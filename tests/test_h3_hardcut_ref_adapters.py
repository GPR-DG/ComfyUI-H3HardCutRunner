"""CPU contract tests for the review-candidate adapter package.

These tests use a tiny fake torch tensor and prove frame/range contracts only;
they are not an RH, CUDA, ComfyUI, H3, or neural execution proof.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np


PLUGIN = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values, normalize_channels=True):
        self.data = np.asarray(values, dtype=np.float32)
        if normalize_channels and self.data.ndim == 4 and self.data.shape[-1] == 1:
            self.data = np.repeat(self.data, 3, axis=-1)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def expand(self, count, *_):
        return Tensor(np.repeat(self.data, count, axis=0))


def load_package():
    missing = object()
    previous_torch = sys.modules.get("torch", missing)
    for name in list(sys.modules):
        if name == "h3_adapter_test_pkg" or name.startswith("h3_adapter_test_pkg."):
            sys.modules.pop(name, None)
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.float32 = np.float32
    torch.cat = lambda chunks, dim=0: Tensor(np.concatenate([x.data for x in chunks], axis=dim))
    sys.modules["torch"] = torch
    spec = importlib.util.spec_from_file_location(
        "h3_adapter_test_pkg", PLUGIN / "h3_hardcut_ref_adapters.py", submodule_search_locations=[str(PLUGIN / "_adapter_test_empty")]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    try:
        spec.loader.exec_module(package)
        sys.modules[spec.name + ".h3_hardcut_ref_adapters"] = package
    finally:
        if previous_torch is missing:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    return package, None if previous_torch is missing else previous_torch


class AdapterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package, cls.previous_torch = load_package()
        cls.mod = sys.modules["h3_adapter_test_pkg.h3_hardcut_ref_adapters"]

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == "h3_adapter_test_pkg" or name.startswith("h3_adapter_test_pkg."):
                sys.modules.pop(name, None)
        if cls.previous_torch is None:
            assert "torch" not in sys.modules
        else:
            assert sys.modules.get("torch") is cls.previous_torch

    def test_registered_adapter_nodes(self):
        for name in ("H3CShotDualRefPad", "H3CShotGuideWindowPlanner", "H3CShotGuideWindowSelect", "H3CShotGuideWindowAppend"):
            node = self.package.NODE_CLASS_MAPPINGS[name]
            self.assertTrue(node.INPUT_TYPES())
            self.assertEqual(len(node.RETURN_TYPES), len(node.RETURN_NAMES))
            self.assertTrue(callable(getattr(node(), node.FUNCTION)))

    def test_dual_ref_padding_holds_current_shot_tail_only(self):
        values = np.arange(14, dtype=np.float32).reshape(14, 1, 1, 1)
        rgb, depth = Tensor(values), Tensor(values + 100)
        manifest = json.dumps({"total_frames": 14, "shots": [
            {"shot": 1, "start": 0, "end": 6, "original_f": 6, "work_l": 39, "seed": 999},
            {"shot": 2, "start": 6, "end": 14, "original_f": 8, "work_l": 39, "seed": 1000},
        ]})
        out = self.mod.H3CShotDualRefPad().run(rgb, depth, manifest, 0)
        raw, rgb_ref, depth_ref = out[:3]
        self.assertEqual((raw.shape[0], rgb_ref.shape[0], depth_ref.shape[0]), (6, 39, 39))
        self.assertEqual(float(rgb_ref.data[-1, 0, 0, 0]), 5.0)
        self.assertEqual(float(depth_ref.data[-1, 0, 0, 0]), 105.0)
        self.assertEqual(out[3:9], (6, 39, 999, 1, "[0,6)", 33))
        out_second = self.mod.H3CShotDualRefPad().run(rgb, depth, manifest, 1)
        self.assertEqual(float(out_second[1].data[-1, 0, 0, 0]), 13.0)
        self.assertEqual(float(out_second[2].data[-1, 0, 0, 0]), 113.0)
        self.assertEqual(out_second[3:9], (8, 39, 1000, 2, "[6,14)", 31))

    def test_dual_ref_requires_per_shot_seed(self):
        image = Tensor(np.zeros((39, 1, 1, 3)))
        manifest = json.dumps({"total_frames": 39, "shots": [{"shot": 1, "start": 0, "end": 39, "original_f": 39, "work_l": 39}]})
        with self.assertRaisesRegex(ValueError, "seed"):
            self.mod.H3CShotDualRefPad().run(image, image, manifest, 0)

    def test_fallback_manifest_matches_production_legal_length_exactly(self):
        self.assertEqual(self.mod._legal_length(40), 56)
        image = Tensor(np.zeros((40, 1, 1, 3)))
        valid = json.dumps({"total_frames": 40, "shots": [{"shot": 1, "start": 0, "end": 40, "original_f": 40, "work_l": 56, "seed": 7}]})
        self.mod.H3CShotDualRefPad().run(image, image, valid, 0)
        for work_l in (39, 73):
            invalid = json.dumps({"total_frames": 40, "shots": [{"shot": 1, "start": 0, "end": 40, "original_f": 40, "work_l": work_l, "seed": 7}]})
            with self.assertRaisesRegex(ValueError, "legal_length|length fields"):
                self.mod.H3CShotDualRefPad().run(image, image, invalid, 0)

    def test_dual_ref_rejects_manifest_count_and_layout_drift(self):
        image = Tensor(np.zeros((10, 1, 1, 3)))
        manifest = json.dumps({"total_frames": 9, "shots": [{"shot": 1, "start": 0, "end": 9, "original_f": 9, "work_l": 39, "seed": 1}]})
        with self.assertRaisesRegex(ValueError, "frame counts"):
            self.mod.H3CShotDualRefPad().run(image, image, manifest, 0)
        manifest = json.dumps({"total_frames": 10, "shots": [{"shot": 1, "start": 0, "end": 10, "original_f": 10, "work_l": 39, "seed": 1}]})
        raw, rgb_ref, depth_ref = self.mod.H3CShotDualRefPad().run(image, Tensor(np.zeros((10, 2, 2, 1))), manifest, 0)[:3]
        self.assertEqual((raw.shape[0], rgb_ref.shape[0], depth_ref.shape[0]), (10, 39, 39))

    def test_dual_ref_allows_aggregate_over_3600(self):
        image = Tensor(np.zeros((3609, 1, 1, 3)))
        manifest = json.dumps({"total_frames": 3609, "shots": [{"shot": 1, "start": 0, "end": 3609, "original_f": 3609, "work_l": 3609, "seed": 1}]})
        out = self.mod.H3CShotDualRefPad().run(image, image, manifest, 0)
        self.assertEqual((out[3], out[4]), (3609, 3609))

    def test_single_generation_still_rejects_over_3600(self):
        with self.assertRaisesRegex(ValueError, "one MiniMax H3 generation"):
            self.mod.H3CShotGuideWindowPlanner().run(3609, 3609, 22)
        with self.assertRaisesRegex(ValueError, "one MiniMax H3 generation"):
            self.mod._require_generation_length(3609, "condition_length")

    def test_short_shot_window_plan_uses_donor_generation_length(self):
        manifest, count, stride, _ = self.mod.H3CShotGuideWindowPlanner().run(39, 124, 22)
        obj = json.loads(manifest)
        self.assertEqual((count, stride), (1, 102))
        self.assertEqual(obj["windows"], [{"window": 1, "start": 0, "source_count": 39, "condition_length": 124, "append_start": 0, "append_count": 39, "guide_frames": 0, "pad_tail": 85}])

    def test_donor_124_102_22_long_shot_contract(self):
        manifest, count, stride, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        obj = json.loads(manifest)
        self.assertEqual((count, stride), (3, 102))
        self.assertEqual([(x["start"], x["source_count"], x["append_start"], x["append_count"], x["guide_frames"], x["pad_tail"]) for x in obj["windows"]], [(0, 124, 0, 124, 0, 0), (102, 124, 22, 102, 22, 0), (204, 39, 22, 17, 22, 85)])
        self.assertEqual(sum(x["append_count"] for x in obj["windows"]), 243)

    def test_window_manifest_rejects_noncanonical_overlap(self):
        manifest = {
            "shot_work_l": 243, "window_size": 124, "overlap": 2, "stride": 122,
            "windows": [
                {"window": 1, "start": 0, "source_count": 124, "condition_length": 124, "append_start": 0, "append_count": 124, "guide_frames": 0, "pad_tail": 0},
                {"window": 2, "start": 122, "source_count": 121, "condition_length": 124, "append_start": 2, "append_count": 119, "guide_frames": 2, "pad_tail": 3},
            ],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.mod._parse_window_manifest(json.dumps(manifest))

    def test_window_manifest_rejects_gap_tamper(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        obj = json.loads(manifest)
        obj["windows"][1]["start"] = 103
        with self.assertRaisesRegex(ValueError, "stride|overlap|append"):
            self.mod._parse_window_manifest(json.dumps(obj))

    def test_window_manifest_rejects_each_boundary_field_tamper(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        cases = (
            ("window", 3),
            ("append_start", 21),
            ("guide_frames", 21),
            ("source_count", 123),
            ("pad_tail", 1),
            ("condition_length", 107),
        )
        for field, value in cases:
            obj = json.loads(manifest)
            obj["windows"][1][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.mod._parse_window_manifest(json.dumps(obj))
        obj = json.loads(manifest)
        obj["window_size"] = 107
        obj["stride"] = 85
        with self.assertRaisesRegex(ValueError, "at least 124"):
            self.mod._parse_window_manifest(json.dumps(obj))

    def test_window_select_is_synchronized_and_tail_padded(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        rgb = Tensor(np.arange(243, dtype=np.float32).reshape(243, 1, 1, 1))
        depth = Tensor((np.arange(243, dtype=np.float32) + 1000).reshape(243, 1, 1, 1))
        out = self.mod.H3CShotGuideWindowSelect().run(rgb, depth, manifest, 2)
        wrgb, wdepth = out[:2]
        self.assertEqual((wrgb.shape[0], wdepth.shape[0]), (124, 124))
        self.assertEqual(float(wrgb.data[0, 0, 0, 0]), 204.0)
        self.assertEqual(float(wrgb.data[38, 0, 0, 0]), 242.0)
        self.assertEqual(float(wrgb.data[-1, 0, 0, 0]), 242.0)
        self.assertEqual(float(wdepth.data[-1, 0, 0, 0]), 1242.0)
        self.assertEqual(out[2:11], (124, 22, 17, 22, 226, 3, True, False, True))

    def test_append_closes_gap_free_generated_window_chain(self):
        planner = self.mod.H3CShotGuideWindowPlanner()
        manifest, _, _, _ = planner.run(243, 124, 22)
        rgb = Tensor(np.arange(243, dtype=np.float32).reshape(243, 1, 1, 1))
        generated = []
        for i in range(3):
            generated.append(self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, i)[0])
        node = self.mod.H3CShotGuideWindowAppend()
        first = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 0)
        acc, count, complete, _ = node.run(generated[0], first[2], first[3], first[4], first[6], 243, first[9], first[10])
        self.assertEqual((acc.shape[0], count, complete), (124, 124, False))
        second = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 1)
        acc, _, complete, _ = node.run(generated[1], second[2], second[3], second[4], second[6], 243, second[9], second[10], acc)
        self.assertEqual((acc.shape[0], complete), (226, False))
        third = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 2)
        acc, _, complete, _ = node.run(generated[2], third[2], third[3], third[4], third[6], 243, third[9], third[10], acc)
        self.assertEqual((acc.shape[0], complete), (243, True))
        self.assertEqual(acc.data[:, 0, 0, 0].tolist(), list(np.arange(243, dtype=np.float32)))


    def test_manifest_and_index_boundaries_are_fail_fast(self):
        image = Tensor(np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1))
        node = self.mod.H3CShotDualRefPad()
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            node.run(image, image, "not-json", 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            node.run(Tensor(np.zeros((0, 1, 1, 1))), Tensor(np.zeros((0, 1, 1, 1))), json.dumps({"total_frames": 0, "shots": []}), 0)
        manifest = json.dumps({"total_frames": 5, "shots": [{"shot": 1, "start": 0, "end": 5, "original_f": 5, "work_l": 39, "seed": 1}]})
        with self.assertRaisesRegex(ValueError, "out of range"):
            node.run(image, image, manifest, 1)

    def test_supported_unique_frame_lengths_and_off_grid_rejection(self):
        lengths = (1, 5, 8, 17, 22, 39, 40, 56, 57, 73, 90, 107, 124, 125, 141, 226, 243, 260, 345)
        node = self.mod.H3CShotDualRefPad()
        for length in lengths:
            work_l = max(39, length)
            work_l += (5 - work_l) % 17
            values = np.arange(length, dtype=np.float32).reshape(length, 1, 1, 1)
            manifest = json.dumps({"total_frames": length, "shots": [{"shot": 1, "start": 0, "end": length, "original_f": length, "work_l": work_l, "seed": 1}]})
            raw, padded = node.run(Tensor(values), Tensor(values + 1000), manifest, 0)[:2]
            self.assertEqual(raw.data[:, 0, 0, 0].tolist(), list(np.arange(length, dtype=np.float32)))
            self.assertEqual(float(padded.data[-1, 0, 0, 0]), float(length - 1))
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            self.mod.H3CShotGuideWindowPlanner().run(40, 124, 22)

    def test_manifest_wrong_order_and_append_reset_are_rejected(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        obj = json.loads(manifest)
        obj["windows"][1]["window"], obj["windows"][2]["window"] = 3, 2
        with self.assertRaisesRegex(ValueError, "ordered"):
            self.mod._parse_window_manifest(json.dumps(obj))
        rgb = Tensor(np.arange(243, dtype=np.float32).reshape(243, 1, 1, 1))
        selected = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 0)
        with self.assertRaisesRegex(ValueError, "without accumulated"):
            self.mod.H3CShotGuideWindowAppend().run(selected[0], selected[2], selected[3], selected[4], selected[6], 243, True, False, selected[0])
    def test_long_aggregate_plans_multiple_legal_generation_windows(self):
        for work_l in (3609, 5003):
            manifest, count, stride, _ = self.mod.H3CShotGuideWindowPlanner().run(work_l, 124, 22)
            obj = json.loads(manifest)
            self.assertGreater(count, 1)
            self.assertEqual(stride, 102)
            self.assertEqual(sum(row["append_count"] for row in obj["windows"]), work_l)
            self.assertTrue(all(5 <= row["condition_length"] <= 3600 and row["condition_length"] % 17 == 5 for row in obj["windows"]))

    def test_append_allows_aggregate_over_3600_but_not_generation_over_3600(self):
        node = self.mod.H3CShotGuideWindowAppend()
        generated = Tensor(np.zeros((124, 1, 1, 3)))
        out = node.run(generated, 124, 0, 124, 0, 3609, True, False)
        self.assertEqual((out[1], out[2]), (124, False))
        with self.assertRaisesRegex(ValueError, "one MiniMax H3 generation"):
            node.run(Tensor(np.zeros((3609, 1, 1, 3))), 3609, 0, 3609, 0, 3609, True, True)

    def test_hard_cut_requires_a_fresh_first_append_state(self):
        node = self.mod.H3CShotGuideWindowAppend()
        generated = Tensor(np.zeros((124, 1, 1, 3)))
        accumulated = node.run(generated, 124, 0, 39, 0, 39, True, True)[0]
        self.assertEqual(accumulated.shape[0], 39)
        fresh = node.run(generated, 124, 0, 39, 0, 39, True, True, None)[0]
        self.assertEqual(fresh.shape[0], 39)

    def test_geometry_contract_allows_same_aspect_ratio_but_rejects_different_ratio(self):
        manifest = json.dumps({"total_frames": 39, "shots": [{"shot": 1, "start": 0, "end": 39, "original_f": 39, "work_l": 39, "seed": 1}]})
        rgb = Tensor(np.zeros((39, 576, 1024, 3)))
        depth = Tensor(np.zeros((39, 288, 512, 3)))
        self.mod.H3CShotDualRefPad().run(rgb, depth, manifest, 0)
        self.mod.H3CShotDualRefPad().run(
            Tensor(np.zeros((39, 736, 1312, 3))),
            Tensor(np.zeros((39, 720, 1280, 3))),
            manifest,
            0,
        )
        with self.assertRaisesRegex(ValueError, "aspect-ratio"):
            self.mod.H3CShotDualRefPad().run(rgb, Tensor(np.zeros((39, 512, 512, 3))), manifest, 0)
        with self.assertRaisesRegex(ValueError, "aspect-ratio"):
            self.mod.H3CShotDualRefPad().run(
                Tensor(np.zeros((39, 448, 880, 3))),
                Tensor(np.zeros((39, 784, 1472, 3))),
                manifest,
                0,
            )

    def test_image_contract_rejects_single_channel_input(self):
        image = Tensor(np.zeros((39, 576, 1024, 1)), normalize_channels=False)
        manifest = json.dumps({"total_frames": 39, "shots": [{"shot": 1, "start": 0, "end": 39, "original_f": 39, "work_l": 39, "seed": 1}]})
        with self.assertRaisesRegex(ValueError, "at least 3 channels"):
            self.mod.H3CShotDualRefPad().run(image, image, manifest, 0)

    def test_append_rejects_subminimum_generation_window(self):
        node = self.mod.H3CShotGuideWindowAppend()
        with self.assertRaisesRegex(ValueError, "at least 124"):
            node.run(Tensor(np.zeros((39, 1, 1, 3))), 39, 0, 39, 0, 39, True, True)

    def test_append_rejects_expected_total_and_accumulated_length_tamper(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        rgb = Tensor(np.arange(243, dtype=np.float32).reshape(243, 1, 1, 1))
        selected = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 0)
        node = self.mod.H3CShotGuideWindowAppend()
        with self.assertRaisesRegex(ValueError, r"17k\+5"):
            node.run(selected[0], selected[2], selected[3], selected[4], selected[6], 242, True, False)
        with self.assertRaisesRegex(ValueError, "accumulated length"):
            node.run(selected[0], selected[2], selected[3], selected[4], 10, 243, False, False, selected[0])

    def test_append_schema_declares_accumulated_optional_and_condition_length_required(self):
        schema = self.mod.H3CShotGuideWindowAppend.INPUT_TYPES()
        self.assertNotIn("accumulated", schema["required"])
        self.assertIn("accumulated", schema["optional"])
        self.assertIn("condition_length", schema["required"])

    def test_ui_minimums_match_runtime_contract(self):
        planner = self.mod.H3CShotGuideWindowPlanner.INPUT_TYPES()["required"]
        self.assertEqual(planner["work_l"][1]["min"], 39)
        self.assertEqual(planner["window_size"][1]["min"], 124)

    def test_planner_rejects_below_39_frame_shot_work_length(self):
        with self.assertRaisesRegex(ValueError, r"17k\+5|39"):
            self.mod.H3CShotGuideWindowPlanner().run(22, 124, 22)

    def test_seed_is_uint64_bounded(self):
        image = Tensor(np.zeros((39, 1, 1, 3)))
        node = self.mod.H3CShotDualRefPad()
        def manifest(seed):
            return json.dumps({"total_frames": 39, "shots": [{"shot": 1, "start": 0, "end": 39, "original_f": 39, "work_l": 39, "seed": seed}]})
        self.assertEqual(node.run(image, image, manifest(0xFFFFFFFFFFFFFFFF), 0)[5], 0xFFFFFFFFFFFFFFFF)
        for seed in (-1, 0x10000000000000000, "not-an-integer", True):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed"):
                    node.run(image, image, manifest(seed), 0)

    def test_short_window_is_padded_to_124_for_generation(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(39, 124, 22)
        rgb = Tensor(np.arange(39, dtype=np.float32).reshape(39, 1, 1, 1))
        out = self.mod.H3CShotGuideWindowSelect().run(rgb, Tensor(np.zeros((39, 2, 2, 1), dtype=np.float32)), manifest, 0)
        self.assertEqual((out[0].shape[0], out[1].shape[0], out[2]), (124, 124, 124))
        self.assertEqual(float(out[0].data[-1, 0, 0, 0]), 38.0)

    def test_append_rejects_generated_length_drift_and_state_mismatch(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(243, 124, 22)
        rgb = Tensor(np.arange(243, dtype=np.float32).reshape(243, 1, 1, 1))
        selected = self.mod.H3CShotGuideWindowSelect().run(rgb, rgb, manifest, 0)
        node = self.mod.H3CShotGuideWindowAppend()
        short = Tensor(np.zeros((123, 1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "condition_length"):
            node.run(short, selected[2], selected[3], selected[4], selected[6], 243, selected[9], selected[10])
        with self.assertRaisesRegex(ValueError, "is_last"):
            node.run(selected[0], selected[2], selected[3], selected[4], selected[6], 124, True, False)
        with self.assertRaisesRegex(ValueError, "is_last"):
            node.run(selected[0], selected[2], selected[3], selected[4], selected[6], 243, True, True)

    def test_rgb_depth_frame_count_still_must_match_but_layout_may_differ(self):
        manifest, _, _, _ = self.mod.H3CShotGuideWindowPlanner().run(39, 124, 22)
        rgb = Tensor(np.zeros((39, 1, 1, 3)))
        depth = Tensor(np.zeros((39, 2, 2, 1)))
        out = self.mod.H3CShotGuideWindowSelect().run(rgb, depth, manifest, 0)
        self.assertEqual((out[0].shape[1:], out[1].shape[1:]), ((1, 1, 3), (2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "frame-count"):
            self.mod.H3CShotGuideWindowSelect().run(rgb, Tensor(np.zeros((38, 2, 2, 1))), manifest, 0)

    def test_fallback_manifest_rejects_noncontiguous_ranges_and_ids(self):
        valid = {"total_frames": 39, "shots": [{"shot": 1, "start": 0, "end": 39, "original_f": 39, "work_l": 39, "seed": 1}]}
        for key, value in (("start", 1), ("shot", 2), ("end", 38)):
            bad = json.loads(json.dumps(valid))
            bad["shots"][0][key] = value
            with self.assertRaises(ValueError):
                self.mod.H3CShotDualRefPad().run(Tensor(np.zeros((39, 1, 1, 3))), Tensor(np.zeros((39, 1, 1, 3))), json.dumps(bad), 0)

if __name__ == "__main__":
    unittest.main()

