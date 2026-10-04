"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy được trên CPU, không cần dữ liệu thật:
    python -m unittest test_code -v        (từ thư mục code/)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

import dataset
import experiments
import inference
import losses
import model as M
import train
from benchmark import bench, latency_report


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits, self.y = torch.randn(32, 9), torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        self.assertLess(abs(fl.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(losses.FocalLoss(2.0)(self.logits, self.y), F.cross_entropy(self.logits, self.y))

    def test_label_smoothing_eps0_equals_ce(self):
        ls = losses.LabelSmoothingCE(0.0)(self.logits, self.y)
        self.assertLess(abs(ls.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_label_smoothing_formula(self):
        eps, k = 0.1, 9
        q = torch.full((32, k), eps / k)
        q[torch.arange(32), self.y] += 1 - eps
        manual = -(q * F.log_softmax(self.logits, 1)).sum(1).mean()
        self.assertLess(abs(losses.LabelSmoothingCE(eps)(self.logits, self.y).item() - manual.item()), 1e-6)

    def test_class_weights(self):
        w = losses.class_weights([100, 100, 400])
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertAlmostEqual((w[0] / w[2]).item(), 4.0, places=4)
        cb = losses.class_weights([100, 100, 400], beta=0.999)
        self.assertGreater(cb[0], cb[2])

    def test_cutmix_lam_matches_box_area_and_labels_mixed(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 3, 32, 32).clone()
        y = torch.arange(8)
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix")
        own = (xm[:, 0] == x[:, 0]).float().mean((1, 2))  # tỉ lệ pixel còn của ảnh gốc
        changed = ya != yb
        np.testing.assert_allclose(own[changed].numpy(), lam, atol=1e-6)
        self.assertTrue(torch.equal(ya, y))
        crit = torch.nn.CrossEntropyLoss()
        logits = torch.randn(8, 9)
        expect = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        self.assertAlmostEqual(losses.mixed_loss(crit, logits, (ya, yb, lam)).item(), expect.item(), places=6)

    def test_mixup(self):
        x, y = torch.randn(4, 3, 8, 8), torch.arange(4)
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 0.4, "mixup")
        perm = [int((yb == i).nonzero()) for i in range(4)]
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[yb]))
        self.assertEqual(sorted(perm), [0, 1, 2, 3])


class TestModel(unittest.TestCase):
    def test_param_groups_and_freeze(self):
        m = M.build_model("resnet10t", pretrained=False)
        groups = {g["name"]: g for g in M.param_groups(m, 1e-4, 1e-3, 0.05)}
        self.assertEqual(set(groups), {"backbone", "backbone_no_wd", "head", "head_no_wd"})
        self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_wd"]["params"]))
        self.assertEqual(groups["backbone_no_wd"]["weight_decay"], 0.0)
        self.assertEqual(groups["head"]["lr"], 1e-3)
        self.assertEqual(sum(p.numel() for g in groups.values() for p in g["params"]),
                         sum(p.numel() for p in m.parameters()))
        M.freeze_backbone(m)
        self.assertEqual({g["name"] for g in M.param_groups(m, 1e-4, 1e-3, 0.05)}, {"head", "head_no_wd"})
        M.set_train_mode(m, frozen=True)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.get_classifier().training)

    def test_vit_pos_embed_has_no_weight_decay(self):
        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=9)
        no_wd = [g for g in M.param_groups(m, 1e-4, 1e-3, 0.05) if g["name"] == "backbone_no_wd"][0]["params"]
        self.assertTrue(any(p is m.pos_embed for p in no_wd))

    def test_counts(self):
        m = timm.create_model("resnet50", pretrained=False, num_classes=1000)
        self.assertAlmostEqual(M.count_params(m), 25.6, delta=0.1)
        self.assertAlmostEqual(M.count_gmacs(m, 224), 4.1, delta=0.1)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_is_exact(self):
        torch.manual_seed(0)
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_large_100", "convnext_atto"):
            m = timm.create_model(name, pretrained=False, num_classes=9).eval()
            for mod in m.modules():  # thống kê BN khác mặc định để phép gộp có ý nghĩa
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.running_mean.uniform_(-0.5, 0.5)
                    mod.running_var.uniform_(0.5, 2.0)
            x = torch.randn(2, 3, 96, 96)
            fused = inference.fuse_conv_bn(m)
            with torch.no_grad():
                err = (m(x) - fused(x)).abs().max().item()
            n_bn = sum(isinstance(mod, torch.nn.BatchNorm2d) for mod in fused.modules())
            self.assertLess(err, 1e-4, name)
            if name != "convnext_atto":
                self.assertGreater(fused.fused_pairs, 0, name)
                self.assertEqual(n_bn, 0, f"{name}: còn {n_bn} BN chưa gộp")

    def test_temperature_recovers_scale(self):
        rng = np.random.default_rng(0)
        true = rng.normal(size=(4000, 9)) * 2
        y = np.array([rng.choice(9, p=p) for p in inference.softmax(true)])
        T = inference.fit_temperature(true * 3, y)  # logit quá tự tin gấp 3 -> T ≈ 3
        self.assertAlmostEqual(T, 3.0, delta=0.3)
        np.testing.assert_array_equal(inference.apply_temperature(true, T).argmax(1), true.argmax(1))

    def test_aggregate_and_views(self):
        z = [np.random.randn(5, 9) for _ in range(3)]
        for space in ("prob", "logit"):
            np.testing.assert_allclose(inference.aggregate_views(z, space).sum(1), 1.0)
        x = torch.randn(2, 3, 256, 256)
        self.assertEqual(len(inference.views_multicrop(x, 224, flip=True)), 10)
        self.assertTrue(torch.equal(inference.view_hflip(inference.view_hflip(x)), x))
        self.assertEqual(inference.views_multiscale(x, [288])[0].shape[-1], 288)

    def test_bench(self):
        r = bench(lambda: sum(range(1000)), warmup=2, iters=20)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        rep = latency_report(timm.create_model("resnet10t", num_classes=9), 1, 64, device="cpu", iters=5)
        self.assertEqual(rep["gpu"], "cpu")


class TestTrainHelpers(unittest.TestCase):
    def test_scheduler_warmup_then_cosine(self):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([{"params": [p], "lr": 1.0}])
        sch = train.build_scheduler(opt, train.Config(epochs=4, warmup_epochs=1), steps_per_epoch=10)
        lrs = []
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        self.assertAlmostEqual(lrs[0], 0.1)
        self.assertAlmostEqual(lrs[9], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(lrs[10:], lrs[11:])))
        self.assertLess(lrs[-1], 0.01)

    def test_ema(self):
        m = torch.nn.Linear(2, 2)
        ema = train.EMA(m, 0.9)
        w0 = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        self.assertTrue(torch.allclose(ema.module.weight, w0 + 0.1))

    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-4"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-4})
        with self.assertRaises(ValueError):
            train.parse_overrides(["khong_co=1"])


class TestEndToEnd(unittest.TestCase):
    """Dataset giả 9 lớp x 12 ảnh, chạy train.run 2 epoch + final_predict + eval + xlsx trên CPU."""

    def test_pipeline(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:  # Windows: memmap còn mở
            d = Path(d)
            (d / "images").mkdir()
            (d / "labels").mkdir()
            rng = np.random.default_rng(0)
            rows = []
            for c in range(9):
                for i in range(12):
                    f = f"c{c}_{i}.jpg"
                    arr = np.clip(rng.normal(30 * c, 20, (256, 256, 3)), 0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(d / "images" / f)
                    rows.append((f, c, dataset.CLASS_NAMES[c]))
            df = pd.DataFrame(rows, columns=["Filename", "Label", "Species"]).sample(frac=1, random_state=0)
            n = len(df)
            parts = {"train": df[: int(0.6 * n)], "val": df[int(0.6 * n): int(0.8 * n)], "test": df[int(0.8 * n):]}
            for k, part in parts.items():
                part.to_csv(d / "labels" / f"{k}_subset0.csv", index=False)
            df.sort_values("Label").to_csv(d / "labels" / "labels.csv", index=False)

            old_total, dataset.TOTAL_IMAGES = dataset.TOTAL_IMAGES, n
            try:
                cache = dataset.build_cache(d / "images", df["Filename"], d / "cache" / "u8.npy")
                ds = dataset.DeepWeedsDataset(parts["val"], d / "images", None, cache)
                np.testing.assert_array_equal(ds[0][0], np.asarray(Image.open(d / "images" / ds[0][2])))
                info = dataset.check_split(*dataset.load_split(d / "labels"), d / "images", verbose=False)
                self.assertEqual(info["union"], n)

                base = dict(backbone="resnet10t", init="scratch", epochs=2, batch_size=16, img_size=64,
                            num_workers=0, amp=False, images_dir=str(d / "images"), labels_dir=str(d / "labels"),
                            cache=str(cache), out_dir=str(d / "runs"), pred_dir=str(d / "pred"),
                            curves_dir=str(d / "curves"))
                for exp, extra in (("T00", {}), ("F01", {"mix": "cutmix", "ema_decay": 0.9, "loss": "focal"})):
                    for seed in (0, 1, 2):
                        s = train.run(train.Config(exp_id=exp, desc="smoke", seed=seed, **base, **extra))
                        self.assertEqual(len(pd.read_csv(d / "runs" / exp / f"seed{seed}" / "history.csv")), 2)
                        cfg = train.load_config(d / "runs", exp, seed)
                        if exp == "T00":
                            experiments.final_predict(cfg, "1view", device="cpu")
                        else:
                            experiments.final_predict(cfg, "hflip", "prob", temperature=True, device="cpu")
                self.assertTrue((d / "curves" / "T00_smoke.png").exists())
                self.assertTrue((d / "pred" / "F01uncal_seed2_test.csv").exists())
                self.assertIn("val_f1_Snake Weed", s)
                again = train.run(train.Config(exp_id="F01", desc="smoke", seed=2, **base))  # đã xong: đọc lại
                self.assertEqual(again["val_macro_f1"], s["val_macro_f1"])
                experiments.alias_run(train.Config(exp_id="T00", desc="smoke", **base), train.Config(exp_id="T99", desc="alias", **base))
                self.assertTrue((d / "curves" / "T99_alias.png").exists())
                self.assertEqual(train.load_config(d / "runs", "T99").exp_id, "T99")
                with self.assertRaises(ValueError):
                    experiments.alias_run(train.Config(exp_id="T00", **base), train.Config(exp_id="T98", **{**base, "aug": "color"}))

                final, per_class = experiments.final_tables(d / "pred", ["F01", "T00"], str(d / "labels/test_subset0.csv"))
                self.assertEqual(len(final), 8)
                self.assertEqual(len(per_class), 18)
                out = experiments.write_results_xlsx(d / "results.xlsx", {"Final": final, "Runs": experiments.collect(d / "runs")},
                                                     {"Runs": "val_macro_f1"})
                self.assertEqual(set(pd.read_excel(out, sheet_name=None)), {"Final", "Runs"})

                import eval as ev
                rc = ev.main(["grade", "--final", str(d / "pred/F01_seed*_test.csv"),
                              "--baseline", str(d / "pred/T00_seed*_test.csv"),
                              "--uncal", str(d / "pred/F01uncal_seed*_test.csv"),
                              "--final-val", str(d / "pred/F01_seed*_val.csv"),
                              "--test-csv", str(d / "labels/test_subset0.csv")])
                self.assertEqual(rc, 0)
            finally:
                dataset.TOTAL_IMAGES = old_total


if __name__ == "__main__":
    unittest.main()
