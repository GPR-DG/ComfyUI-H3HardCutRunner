"""Contract tests for C's observable, per-shot modular path (no RH generation)."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


PLUGIN = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("H3_C_PROJECT_ROOT", PLUGIN.parents[2])).resolve()
WORKFLOW = PROJECT_ROOT / "outputs" / "H3_V16_hardcut_runner_RH_SCENE_VLM_C.json"
A2_WORKFLOW = PROJECT_ROOT / "outputs" / "H3_V16_hardcut_runner_RH_SCENE_VLM_C_A2_FORCE_REFS.json"
BASELINE = PROJECT_ROOT / "work" / "modular_c_baseline_20261001" / WORKFLOW.name


class Tensor:
    def __init__(self, values):
        self.data = np.asarray(values, dtype=np.float32)

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

    def detach(self):
        return self

    def to(self, **_):
        return self

    def copy_(self, other):
        self.data[...] = other.data
        return self


def load_package():
    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.float32 = np.float32
    torch.cat = lambda chunks, dim=0: Tensor(np.concatenate([x.data for x in chunks], axis=dim))
    torch.empty = lambda shape, **_: Tensor(np.empty(shape, dtype=np.float32))
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    sys.modules["torch"] = torch
    sys.modules["nodes"] = nodes
    spec = importlib.util.spec_from_file_location(
        "h3_modular_test_pkg", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


class ModularCContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host_modules = patch.dict(sys.modules)
        host_modules.start()
        cls.addClassCleanup(host_modules.stop)
        cls.package = load_package()

    def test_registered_nodes_have_comfy_contracts(self):
        for name in (
            "H3ShotSceneVLM", "H3OptionalPicture3",
            "H3CShotPlanner", "H3CShotSelectPad", "H3CShotPolicy",
            "H3CPromptCompiler", "H3CShotTrim", "H3COrderedMergeStep",
            "H3COrderedMergeFinish",
        ):
            node = self.package.NODE_CLASS_MAPPINGS[name]
            self.assertTrue(node.INPUT_TYPES())
            self.assertTrue(node.RETURN_TYPES)
            self.assertTrue(node.FUNCTION)
            self.assertTrue(node.CATEGORY)
            self.assertTrue(callable(getattr(node(), node.FUNCTION)))
            self.assertEqual(len(node.RETURN_TYPES), len(node.RETURN_NAMES))

    def test_manifest_preserves_zero_based_seed_and_legal_length(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        old = mod._detect_hard_cuts
        mod._detect_hard_cuts = lambda _rgb, _threshold, _minimum: [(0, 10), (10, 27)]
        try:
            image = Tensor(np.zeros((27, 2, 2, 3)))
            manifest, count, total, _ = mod.H3CShotPlanner().run(image, image, 999, .18, 8)
        finally:
            mod._detect_hard_cuts = old
        shots = json.loads(manifest)["shots"]
        self.assertEqual((count, total), (2, 27))
        self.assertEqual([(s["shot"], s["start"], s["end"], s["original_f"], s["work_l"], s["seed"]) for s in shots],
                         [(1, 0, 10, 10, 39, 999), (2, 10, 27, 17, 39, 1000)])

    def test_select_depth_uses_same_range_and_last_frame_padding(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        values = np.arange(27, dtype=np.float32).reshape(27, 1, 1, 1)
        image = Tensor(values)
        manifest = json.dumps({"total_frames": 27, "shots": [
            {"shot": 1, "start": 0, "end": 10, "original_f": 10, "work_l": 39, "seed": 999},
            {"shot": 2, "start": 10, "end": 27, "original_f": 17, "work_l": 39, "seed": 1000},
        ]})
        rgb, depth, original_f, work_l, seed, number, _ = mod.H3CShotSelectPad().run(image, image, manifest, 1)
        self.assertEqual((rgb.shape[0], depth.shape[0], original_f, work_l, seed, number), (17, 39, 17, 39, 1000, 2))
        self.assertEqual(float(depth.data[-1, 0, 0, 0]), 26)

    def test_policy_and_prompt_preserve_reference_and_empty_contract(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = "\n".join(json.dumps(x) for x in (
            {"shot": 1, "prompt": "Scene: room", "subject_mode": "present", "secondary_human_presence": "none", "scene_fields": {"scene": "room"}},
            {"shot": 2, "prompt": "Scene: hall", "subject_mode": "absent", "secondary_human_presence": "none", "scene_fields": {"scene": "hall"}},
        ))
        picture = Tensor(np.zeros((1, 1, 1, 3)))
        outputs = mod.H3CShotPolicy().run(rows, "2", 1, 2, 999, picture, picture, picture3=None, manual_force_empty_indices="")
        self.assertIs(outputs[0], picture)
        self.assertIs(outputs[1], picture)
        self.assertIsNone(outputs[2])
        self.assertEqual(outputs[3:6], (False, False, True))
        self.assertEqual(outputs[12:16], (True, True, False, 999))
        prompts = mod.H3CPromptCompiler().run(rows, 1, "A [[PICTURE3_CONTRACT_BEGIN]]rear[[PICTURE3_CONTRACT_END]] B", "second", False, False, False)
        self.assertNotIn("rear", prompts[0])
        self.assertIn("Scene: room", prompts[0])
        self.assertIn('"scene": "room"', prompts[2])
        empty = mod.H3CShotPolicy().run(rows, "2", 2, 2, 1000, picture, picture, picture3=picture, manual_force_empty_indices="")
        self.assertEqual(empty[0:3], (None, None, None))
        self.assertEqual(empty[3:6], (True, False, False))
        self.assertEqual(empty[12:16], (False, False, False, 1000))
        with self.assertRaises(ValueError):
            mod.H3CShotPolicy().run(rows, "2", 1, 3, 999, picture, picture)

    def test_force_present_overrides_auto_empty_and_cleans_both_prompts(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        from h3_modular_test_pkg import h3_hardcut_runner_c as c
        fields = {
            "scene": "atrium", "space_structure": "arched corridor",
            "environment_objects": "wooden bench",
            "foreground_midground_background": "bench in foreground",
            "materials_colors": "brown stone", "lighting": "soft daylight",
            "color_temperature_palette": "warm amber", "composition_camera": "tracking camera",
            "scene_motion": "curtains moving", "temporal_structure": "slow pan",
        }
        raw = {"subject_mode": "absent", "secondary_human_presence": "present", **fields}
        record = c._normalise_scene_record(json.dumps(raw), 1)
        rows = json.dumps({"shot": 1, **record})
        picture = Tensor(np.ones((1, 1, 1, 3)))
        policy = mod.H3CShotPolicy().run(rows, "1", 1, 1, 999, picture, picture,
                                         picture3=picture, manual_force_present_indices="1")
        self.assertEqual(policy[3:6], (False, False, True))
        self.assertEqual(policy[12:16], (True, True, True, 999))
        self.assertIn("manual_force_present", policy[10])
        prompts = mod.H3CPromptCompiler().run(rows, 1, "first global", "second global",
                                               policy[3], policy[4], policy[11])
        for prompt in prompts[:2]:
            for value in fields.values():
                self.assertIn(value, prompt)
            for forbidden in ("No person is visible", "source primary protagonist absent",
                              "Secondary-human policy"):
                self.assertNotIn(forbidden, prompt)
        self.assertEqual(json.loads(prompts[2]), fields)

    def test_force_present_secondary_uncertain_and_picture3_optional(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = json.dumps({"shot": 1, "prompt": "No person is visible as the source primary protagonist; "
                            "preserve only source-visible secondary/background humans without injecting the target identity.\n"
                            "Scene: station", "subject_mode": "absent",
                            "secondary_human_presence": "uncertain", "scene_fields": {"scene": "station"}})
        picture = Tensor(np.ones((1, 1, 1, 3)))
        policy = mod.H3CShotPolicy().run(rows, "", 1, 1, 999, picture, picture,
                                         picture3=None, manual_force_present_indices="1")
        self.assertEqual(policy[3:6], (False, False, True))
        self.assertEqual(policy[12:16], (True, True, False, 999))
        prompt = mod.H3CPromptCompiler().run(rows, 1, "first", "second", *policy[3:5], False)[0]
        self.assertIn("station", prompt)
        self.assertNotIn("No person is visible", prompt)

    def test_forced_primary_keeps_secondary_people_separate(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        from h3_modular_test_pkg import h3_hardcut_runner_c as c
        picture = Tensor(np.ones((1, 1, 1, 3)))
        for secondary in ("present", "uncertain"):
            with self.subTest(secondary=secondary):
                source_present = c._normalise_scene_record(json.dumps({
                    "subject_mode": "present", "secondary_human_presence": secondary, "scene": "hall"
                }), 1)
                misclassified = c._normalise_scene_record(json.dumps({
                    "subject_mode": "absent", "secondary_human_presence": secondary, "scene": "hall"
                }), 1)
                normal_prompt = mod.H3CPromptCompiler().run(
                    json.dumps({"shot": 1, **source_present}), 1, "first", "second", False, False, False)[0]
                rows = json.dumps({"shot": 1, **misclassified})
                policy = mod.H3CShotPolicy().run(rows, "", 1, 1, 999, picture, picture,
                                                 manual_force_present_indices="1")
                self.assertEqual(policy[3:6], (False, False, True))
                forced_prompts = mod.H3CPromptCompiler().run(rows, 1, "first", "second", *policy[3:5], False)
                protection = "preserve source-visible secondary/background humans as separate non-target people"
                self.assertIn(protection, normal_prompt)
                for prompt in forced_prompts[:2]:
                    self.assertIn(protection, prompt)
                    self.assertNotIn("source primary protagonist absent", prompt)

    def test_policy_report_distinguishes_source_and_effective_presence(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = json.dumps({"shot": 1, "prompt": "Scene: hall", "subject_mode": "absent"})
        picture = Tensor(np.ones((1, 1, 1, 3)))
        report = mod.H3CShotPolicy().run(rows, "1", 1, 1, 999, picture, picture,
                                         manual_force_present_indices="1")[10]
        self.assertIn("source_subject_mode=absent", report)
        self.assertIn("effective_subject_mode=present", report)

    def test_cleanup_never_silently_deletes_multiline_scene_content(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        from h3_modular_test_pkg import h3_hardcut_runner_c as c
        record = c._normalise_scene_record(json.dumps({
            "subject_mode": "absent", "secondary_human_presence": "none",
            "scene": "hall\nNo person is visible on a poster", "lighting": "warm"
        }), 1)
        with self.assertRaisesRegex(ValueError, "contradictory absent-person policy"):
            mod.H3CPromptCompiler().run(json.dumps({"shot": 1, **record}), 1,
                                         "first", "second", False, False, False)

    def test_force_present_absent_none_and_existing_present_remain_distinct(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        from h3_modular_test_pkg import h3_hardcut_runner_c as c
        records = [c._normalise_scene_record(json.dumps({
            "subject_mode": mode, "secondary_human_presence": "none", "scene": "lit courtyard"
        }), number) for number, mode in ((1, "absent"), (2, "present"))]
        rows = "\n".join(json.dumps({"shot": i, **record}) for i, record in enumerate(records, 1))
        picture = Tensor(np.ones((1, 1, 1, 3)))
        forced = mod.H3CShotPolicy().run(rows, "1", 1, 2, 999, picture, picture,
                                         manual_force_present_indices="1")
        ordinary = mod.H3CShotPolicy().run(rows, "1", 2, 2, 1000, picture, picture)
        self.assertEqual(forced[3:6], ordinary[3:6])
        self.assertEqual(forced[12:15], ordinary[12:15])
        prompt = mod.H3CPromptCompiler().run(rows, 1, "first", "second", *forced[3:5], False)[0]
        self.assertIn("lit courtyard", prompt)
        self.assertNotIn("No person is visible", prompt)

    def test_force_present_fails_closed_on_contradictory_scene_field(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = json.dumps({"shot": 1, "prompt": "No person is visible in this shot; preserve only the environment.\n"
                           "Scene: primary protagonist absent behind a wall",
                           "subject_mode": "absent", "secondary_human_presence": "none",
                           "scene_fields": {"scene": "primary protagonist absent behind a wall"}})
        with self.assertRaisesRegex(ValueError, "contradictory absent-person policy"):
            mod.H3CPromptCompiler().run(rows, 1, "first", "second", False, False, False)

    def test_force_present_conflicts_with_explicit_empty_but_not_auto_empty(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = json.dumps({"shot": 1, "prompt": "Scene: room", "subject_mode": "absent"})
        picture = Tensor(np.ones((1, 1, 1, 3)))
        with self.assertRaisesRegex(ValueError, "manual_force_empty_indices.*manual_force_present_indices"):
            mod.H3CShotPolicy().run(rows, "1", 1, 1, 999, picture, picture,
                                    manual_force_empty_indices="1", manual_force_present_indices="1")
        with self.assertRaisesRegex(ValueError, "manual_force_present_indices"):
            mod.H3CShotPolicy().run(rows, "", 1, 1, 999, picture, picture,
                                    manual_force_present_indices="2")
        old = mod.H3CShotPolicy().run(rows, "1", 1, 1, 999, picture, picture)
        blank = mod.H3CShotPolicy().run(rows, "1", 1, 1, 999, picture, picture,
                                        manual_force_present_indices="")
        self.assertEqual(old, blank)
        self.assertTrue(old[3])

    def test_force_present_leaves_other_empty_shots_unchanged(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        rows = "\n".join(json.dumps({"shot": i, "prompt": f"Scene: {i}",
                                      "subject_mode": "absent"}) for i in range(1, 7))
        picture = Tensor(np.ones((1, 1, 1, 3)))
        policy = mod.H3CShotPolicy().run(rows, "1,2,3,4,5,6", 6, 6, 1004, picture, picture,
                                         manual_force_present_indices="1,2,3,4,5")
        self.assertEqual(policy[3:6], (True, False, False))
        self.assertEqual(policy[0:3], (None, None, None))

    def test_trim_and_ordered_merge_enforce_frame_contract(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        trim = mod.H3CShotTrim()
        step = mod.H3COrderedMergeStep()
        first = trim.run(Tensor(np.ones((39, 1, 1, 3))), 10, 39, 1)[0]
        second = trim.run(Tensor(np.full((39, 1, 1, 3), 2)), 17, 39, 2)[0]
        with self.assertRaises(ValueError):
            trim.run(Tensor(np.ones((38, 1, 1, 3))), 10, 39, 1)
        state = step.run(None, first, 1, 27, "[0,10)", "policy1", "prompt1", "shot1")[0]
        with self.assertRaises(ValueError):
            step.run(state, second, 3, 27, "[10,27)", "policy3", "prompt3", "bad order")
        state = step.run(state, second, 2, 27, "[10,27)", "policy2", "prompt2", "shot2")[0]
        images, fps, count, report = mod.H3COrderedMergeFinish().run(state, 24.0, 27)
        self.assertEqual((images.shape[0], fps, count), (27, 24.0, 27))
        self.assertEqual(float(images.data[0, 0, 0, 0]), 1)
        self.assertEqual(float(images.data[-1, 0, 0, 0]), 2)
        self.assertIn("shot2", report)
        self.assertIn("policy2", report)
        self.assertIn("prompt2", report)

    def test_three_shot_dynamic_dispatch_policy_and_ordered_merge_simulation(self):
        from h3_modular_test_pkg import h3_hardcut_modular_c as mod
        old = mod._detect_hard_cuts
        mod._detect_hard_cuts = lambda *_: [(0, 6), (6, 14), (14, 24)]
        video = Tensor(np.arange(24, dtype=np.float32).reshape(24, 1, 1, 1))
        try:
            manifest, shot_count, total, _ = mod.H3CShotPlanner().run(video, video, 999, .18, 8)
        finally:
            mod._detect_hard_cuts = old
        rows = "\n".join(json.dumps(record) for record in (
            {"shot": 1, "prompt": "room", "subject_mode": "present", "secondary_human_presence": "none"},
            {"shot": 2, "prompt": "street", "subject_mode": "absent", "secondary_human_presence": "present"},
            {"shot": 3, "prompt": "park", "subject_mode": "absent", "secondary_human_presence": "none"},
        ))
        picture = Tensor(np.ones((1, 1, 1, 3)))
        state = None
        observed = []
        for index in range(shot_count):
            rgb, depth, original_f, work_l, seed, number, _ = mod.H3CShotSelectPad().run(video, video, manifest, index)
            policy = mod.H3CShotPolicy().run(rows, "3", number, shot_count, seed, picture, picture, picture3=picture)
            observed.append((number, original_f, work_l, seed, policy[3], policy[4], policy[12:15]))
            if index == 1:
                self.assertEqual(policy[0:3], (None, None, None))
            self.assertEqual(rgb.shape[0], original_f)
            self.assertEqual(depth.shape[0], work_l)
            decoded = Tensor(np.full((work_l, 1, 1, 3), number))
            trimmed, count, report = mod.H3CShotTrim().run(decoded, original_f, work_l, number)
            self.assertEqual(count, original_f)
            state = mod.H3COrderedMergeStep().run(state, trimmed, number, total, f"[{index},{index+1})",
                                                   policy[10], f"prompt{number}", report)[0]
        merged, fps, count, _ = mod.H3COrderedMergeFinish().run(state, 24.0, total)
        self.assertEqual(count, 24)
        self.assertEqual(fps, 24.0)
        self.assertEqual(observed, [
            (1, 6, 39, 999, False, False, (True, True, True)),
            (2, 8, 39, 1000, False, True, (False, False, False)),
            (3, 10, 39, 1001, True, False, (False, False, False)),
        ])
        self.assertEqual(merged.data[:, 0, 0, 0].tolist(), [1.0] * 6 + [2.0] * 8 + [3.0] * 10)

    def test_scene_vlm_exposes_raw_normalised_and_policy_fields(self):
        from h3_modular_test_pkg import h3_hardcut_runner_c as c
        old_detect, old_invoker = c._detect_hard_cuts, c._SceneVLMInvoker
        class FakeInvoker:
            def __call__(self, _values, **_kwargs):
                return (json.dumps({"subject_mode": "present", "secondary_human_presence": "present",
                                    "scene": "red room", "subject_associated_prop_policy": "present"}),)
            def report_snapshot(self):
                return {"node_class": "AILab_QwenVL_Advanced"}
        c._detect_hard_cuts = lambda *_: [(0, 5)]
        c._SceneVLMInvoker = FakeInvoker
        try:
            out = c.H3ShotSceneVLM().run(Tensor(np.zeros((5, 1, 1, 3))), "Qwen3-VL-8B-Instruct",
                                         "4-bit (VRAM-friendly)", "auto", False, "auto", 1024, 32,
                                         "auto", 1, 0.18, 8)
        finally:
            c._detect_hard_cuts, c._SceneVLMInvoker = old_detect, old_invoker
        self.assertEqual(len(out), 11)  # orientation is append-only; old slots stay 0..9
        raw = [json.loads(line) for line in out[3].splitlines()]
        self.assertEqual(raw[0]["shot"], 1)
        self.assertIn('"scene": "red room"', raw[0]["raw"])
        self.assertIn('"shot":1', out[4])
        self.assertIn("red room", out[5])
        self.assertEqual(json.loads(out[6]), {"shot": 1, "value": "present"})
        self.assertEqual(json.loads(out[7]), {"shot": 1, "value": "present"})
        self.assertEqual(json.loads(out[8]), {"shot": 1, "value": "present"})
        self.assertIn("AILab_QwenVL_Advanced", out[9])
        self.assertEqual(json.loads(out[10])["value"], "uncertain")

    def test_production_json_has_modular_ancestry_and_no_legacy_runner(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        by_target = {}
        for link in workflow["links"]:
            self.assertIn(link[1], nodes)
            self.assertIn(link[3], nodes)
            by_target.setdefault(link[3], []).append(link)
        save = next(node for node in nodes.values() if node["type"] == "SaveVideo" and node["mode"] == 0)
        ancestors = set()
        def walk(node_id):
            if node_id in ancestors:
                return
            ancestors.add(node_id)
            for link in by_target.get(node_id, []):
                walk(link[1])
        walk(save["id"])
        types = {nodes[node_id]["type"] for node_id in ancestors}
        self.assertNotIn("H3HardCutRunnerSceneVLM", types)
        self.assertTrue({"H3CShotPlanner", "H3CShotSelectPad", "H3CShotPolicy", "H3CPromptCompiler", "H3CShotTrim", "H3COrderedMergeFinish", "MiniMaxH3ReferenceToVideo", "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVConcatAVLatent", "MinimaxH3LatentUpscaler3D", "VAEDecode"} <= types)

    def test_metadata_matches_actual_modular_nodes(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        types = {node["type"] for node in workflow["nodes"]}
        metadata = workflow["extra"]["h3_scene_vlm_candidate"]
        self.assertEqual(set(metadata["active_nodes"]), {
            "H3ShotSceneVLM", "H3OptionalPicture3", "H3CShotPlanner",
            "H3CShotSelectPad", "H3CShotPolicy", "H3CPromptCompiler",
            "H3CShotTrim", "H3COrderedMergeStep", "H3COrderedMergeFinish",
        })
        self.assertTrue(set(metadata["active_nodes"]) <= types)
        self.assertNotIn("H3HardCutRunnerSceneVLM", types)
        self.assertFalse(any(node.get("properties", {}).get("cnr_id") == "local-h3-hardcut"
                             for node in workflow["nodes"] if node["type"].startswith("H3C")))

    def test_every_link_and_new_node_port_has_a_matching_contract(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(len(nodes), len(workflow["nodes"]))
        links = {link[0]: link for link in workflow["links"]}
        self.assertEqual(len(links), len(workflow["links"]))
        self.assertEqual(workflow["last_link_id"], len(links))
        for link_id, (_, source, out_slot, target, in_slot, kind) in links.items():
            output = nodes[source]["outputs"][out_slot]
            input_ = nodes[target]["inputs"][in_slot]
            self.assertIn(link_id, output["links"])
            self.assertEqual(input_["link"], link_id)
            self.assertIn(kind, (output["type"], "*"))
            self.assertTrue(input_["type"] == output["type"] or "*" in (input_["type"], output["type"]))
        for node in nodes.values():
            for input_ in node["inputs"]:
                if input_["link"] is not None:
                    self.assertIn(input_["link"], links)
            for output in node["outputs"]:
                for link_id in output["links"] or []:
                    self.assertIn(link_id, links)
            if node["type"].startswith("H3C"):
                klass = self.package.NODE_CLASS_MAPPINGS[node["type"]]
                schema = klass.INPUT_TYPES()
                fields = {**schema.get("required", {}), **schema.get("optional", {})}
                self.assertEqual({x["name"] for x in node["inputs"]}, set(fields))
                self.assertEqual([x["name"] for x in node["outputs"]], list(klass.RETURN_NAMES))
            elif node["type"] == "H3ShotSceneVLM":
                # Legacy canvases can store the old output prefix; no slot may move.
                klass = self.package.NODE_CLASS_MAPPINGS[node["type"]]
                count = len(node["outputs"])
                self.assertLessEqual(count, len(klass.RETURN_NAMES))
                self.assertEqual([x["name"] for x in node["outputs"]], list(klass.RETURN_NAMES[:count]))
                self.assertEqual([x["type"] for x in node["outputs"]], list(klass.RETURN_TYPES[:count]))
        self.assertEqual(sum(1 for x in nodes.values() if x["type"] == "SaveVideo"), 1)
        self.assertEqual(sum(1 for x in nodes.values() if x["type"] == "H3HardCutRunnerSceneVLM"), 0)

    def test_source_settings_and_two_pass_order_match_old_c(self):
        old = {n["id"]: n for n in json.loads(BASELINE.read_text(encoding="utf-8"))["nodes"]}
        new = {n["id"]: n for n in json.loads(WORKFLOW.read_text(encoding="utf-8"))["nodes"]}
        for node_id in (8, 9, 12, 22, 33, 619, 620, 622, 626, 627, 631, 632, 633,
                        634, 635, 643, 645, 700, 702, 703, 704, 707, 712, 713):
            self.assertEqual(new[node_id]["type"], old[node_id]["type"])
            self.assertEqual(new[node_id].get("widgets_values"), old[node_id].get("widgets_values"))
        runner_widgets = old[701]["widgets_values"]
        self.assertEqual(new[714]["widgets_values"][0], runner_widgets[7])  # shot seed
        self.assertEqual(new[720]["widgets_values"][0], runner_widgets[11])  # fps
        self.assertEqual(new[642]["widgets_values"], runner_widgets[6:7] + ["target dimensions"] + runner_widgets[4:6] + runner_widgets[12:16])
        self.assertEqual(new[629]["widgets_values"][-1], "match")
        self.assertEqual(new[646]["widgets_values"][-1], "match")
        self.assertEqual(new[713]["type"], "H3OptionalPicture3")
        self.assertIn("IMAGEUPLOAD", {x["type"] for x in new[713]["inputs"]})

    def test_all_nodes_contribute_to_a_formal_output_or_debug_sink(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        parents = {}
        for link in workflow["links"]:
            parents.setdefault(link[3], set()).add(link[1])
        sinks = [n["id"] for n in nodes.values() if n["type"] in {"SaveVideo", "PreviewImage", "PreviewAny"}]
        reached = set()
        frontier = list(sinks)
        while frontier:
            current = frontier.pop()
            if current not in reached:
                reached.add(current)
                frontier.extend(parents.get(current, ()))
        self.assertEqual(reached, set(nodes))

    def test_critical_two_pass_and_loop_wiring(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in workflow["nodes"]}
        edges = {
            (source, nodes[source]["outputs"][out_slot]["name"], target,
             nodes[target]["inputs"][in_slot]["name"])
            for _, source, out_slot, target, in_slot, _ in workflow["links"]
        }
        must = {
            (714, "shot_count", 721, "total"),
            (721, "index", 715, "index"),
            (715, "depth_ref_video", 629, "ref_videos.ref_video_0"),
            (717, "first_pass_prompt", 629, "prompt"),
            (717, "second_pass_prompt", 646, "prompt"),
            (635, "high_sigmas", 624, "sigmas"),
            (635, "low_sigmas", 636, "sigmas"),
            (624, "denoised_output", 637, "av_latent"),
            (637, "video_latent", 642, "latent"),
            (637, "audio_latent", 638, "audio_latent"),
            (642, "latent", 638, "video_latent"),
            (638, "latent", 636, "latent_image"),
            (636, "output", 639, "samples"),
            (639, "IMAGE", 718, "decoded"),
            (715, "work_l", 718, "work_l"),
            (721, "value1", 719, "previous"),
            (719, "state", 722, "initial_value1"),
            (722, "value1", 720, "state"),
            (720, "images", 704, "images"),
            (720, "fps", 704, "fps"),
            (704, "VIDEO", 33, "video"),
        }
        self.assertTrue(must <= edges, sorted(must - edges))
        self.assertFalse(any(s == 646 and t == 629 for s, _, t, _ in edges))
        self.assertFalse(any(s == 715 and sn == "depth_ref_video" and t == 646 for s, sn, t, _ in edges))
        for h3 in (629, 646):
            for index in range(3):
                self.assertIn((716, f"picture{index + 1}", h3, f"ref_images.ref_image_{index}"), edges)
        self.assertIn((624, "denoised_output", 724, "samples"), edges)
        self.assertIn((724, "IMAGE", 725, "images"), edges)

    def test_a2_builder_is_copy_only_and_preserves_dynamic_vlm_and_links(self):
        spec = importlib.util.spec_from_file_location("h3_a2_builder", PLUGIN / "tools" / "build_h3_modular_c.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        before = WORKFLOW.read_bytes()
        old = json.loads(before)
        self.assertEqual(builder.build_a2_force_refs(), (len(old["nodes"]), len(old["links"])))
        self.assertEqual(WORKFLOW.read_bytes(), before)
        first_sha = hashlib.sha256(A2_WORKFLOW.read_bytes()).hexdigest()
        builder.build_a2_force_refs()
        self.assertEqual(hashlib.sha256(A2_WORKFLOW.read_bytes()).hexdigest(), first_sha)
        new = json.loads(A2_WORKFLOW.read_text(encoding="utf-8"))
        for key in old.keys() - {"nodes", "extra"}:
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["links"], old["links"])
        self.assertEqual(len(new["nodes"]), len(old["nodes"]))
        old_nodes = {n["id"]: n for n in old["nodes"]}
        new_nodes = {n["id"]: n for n in new["nodes"]}
        self.assertEqual(set(new_nodes), set(old_nodes))
        for node_id in old_nodes.keys() - {700, 716}:
            self.assertEqual(new_nodes[node_id], old_nodes[node_id])
        self.assertEqual({k:v for k,v in new_nodes[700].items() if k != "outputs"},
                         {k:v for k,v in old_nodes[700].items() if k != "outputs"})
        self.assertEqual(new_nodes[700]["outputs"][:len(old_nodes[700]["outputs"])], old_nodes[700]["outputs"])
        self.assertEqual([x["name"] for x in new_nodes[700]["outputs"]],
                         list(self.package.NODE_CLASS_MAPPINGS["H3ShotSceneVLM"].RETURN_NAMES))
        policy = new_nodes[716]
        self.assertEqual(policy["inputs"], old_nodes[716]["inputs"])
        self.assertEqual(old_nodes[716]["widgets_values"], ["", ""])
        self.assertEqual([x["name"] for x in policy["inputs"]],
                         list(self.package.NODE_CLASS_MAPPINGS["H3CShotPolicy"].INPUT_TYPES()["required"]) +
                         list(self.package.NODE_CLASS_MAPPINGS["H3CShotPolicy"].INPUT_TYPES()["optional"]))
        self.assertEqual(builder.A2_FORCE_PRESENT, "1,9")
        self.assertEqual(policy["widgets_values"], ["", "1,9"])
        self.assertEqual(policy["inputs"][-1]["link"], None)
        self.assertEqual(new_nodes[717], old_nodes[717])
        self.assertTrue(any(link[1] == 700 and link[3] == 717 and
                            new_nodes[700]["outputs"][link[2]]["name"] == "shot_prompts"
                            for link in new["links"]))


if __name__ == "__main__":
    unittest.main()
