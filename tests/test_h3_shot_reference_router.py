"""CPU logic tests; no Qwen, H3, ComfyUI scheduler or GPU inference is run.

NumPy images are passed through by identity. Only unavailable host imports and
the external Qwen engine are stubbed; routing, parsers, policy and VLM dispatch
are the real plugin code. sys.modules is restored after each test.
"""
from __future__ import annotations

import importlib.util
import copy
import ast
import json
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

PLUGIN = Path(__file__).resolve().parents[1]
ORIENTATIONS = ("front", "back", "side", "mixed", "turning", "uncertain")
OLD_VLM_OUTPUTS = ("shot_prompts", "empty_shot_indices", "report", "raw_vlm_output",
                   "normalised_vlm_output", "scene_fields", "subject_mode",
                   "secondary_human_presence", "source_prop_policy", "vlm_runtime")


@contextmanager
def package_context():
    torch = types.ModuleType("torch")
    torch.Tensor = np.ndarray
    torch.float32 = np.float32
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    with patch.dict(sys.modules, {"torch": torch, "nodes": nodes}):
        spec = importlib.util.spec_from_file_location(
            "_h3_routing_test", PLUGIN / "__init__.py",
            submodule_search_locations=[str(PLUGIN)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pkg
        spec.loader.exec_module(pkg)
        yield pkg


def scene_rows(orientation="front", mode="present", secondary="none", shot=1):
    return json.dumps({"shot": shot, "prompt": "Scene: courtyard",
                       "subject_mode": mode, "secondary_human_presence": secondary,
                       "orientation": orientation, "orientation_reason": "torso visible",
                       "orientation_evidence": {"begin": orientation, "middle": orientation,
                           "end": orientation, "all_sampled_frames_same": orientation in ("front", "back")}})


class CPUImage(np.ndarray):
    """NumPy seam for the unchanged detector/pad/merge code; not real torch."""
    def abs(self):
        return np.abs(self)
    def mean(self, dim):
        return np.asarray(self).mean(axis=dim).view(CPUImage)
    def detach(self):
        return self
    def cpu(self):
        return self
    def float(self):
        return self.astype(np.float32)
    def expand(self, count, *_):
        return np.repeat(self, count, axis=0)
    def to(self, *, device, dtype):
        return self.astype(dtype, copy=False)
    def copy_(self, other):
        self[...] = other
        return self


class ShotReferenceTests(unittest.TestCase):
    def setUp(self):
        self.context = package_context()
        self.pkg = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.c = sys.modules["_h3_routing_test.h3_hardcut_runner_c"]
        self.mod = sys.modules["_h3_routing_test.h3_hardcut_modular_c"]
        self.front = [np.full((1, 2, 3, 3), i, np.float32) for i in range(1, 5)]
        self.back = [np.full((1, 3, 2, 3), i, np.float32) for i in range(5, 9)]

    def route(self, orientation="front", inject=True, back=False, rows=None, router=None, **kwargs):
        cls = self.pkg.NODE_CLASS_MAPPINGS.get("H3CShotReferenceRouter")
        self.assertIsNotNone(cls, "new router must be registered via __init__.py")
        values = dict(shot_prompts=rows or scene_rows(orientation), shot_number=1,
                      inject_target_references=inject, front_full=self.front[0],
                      front_detail_1=self.front[1], front_detail_2=self.front[2],
                      front_detail_3=self.front[3])
        if back:
            values.update(back_full=self.back[0], back_detail_1=self.back[1],
                          back_detail_2=self.back[2], back_detail_3=self.back[3])
        values.update(kwargs)
        return (router or cls()).run(**values)

    def assert_images(self, out, expected):
        self.assertEqual(len(out), 12)
        for actual, want in zip(out[:8], [*expected, *([None] * (8-len(expected)))]):
            self.assertIs(actual, want)
        self.assertEqual(out[9], json.loads(out[11])["orientation"])
        mapping = json.loads(out[11])["references"]
        self.assertEqual(len(mapping), len(expected))
        for i, row in enumerate(mapping):
            self.assertEqual(row["slot"], f"ref_image_{i}")
            self.assertEqual(row["picture"], i+1)
            self.assertIn(f"<Picture {i+1}>", out[10])
        self.assertNotIn("<Picture 9>", out[10])

    def test_single_front_fallback_for_all_six_orientations(self):
        for orientation in ORIENTATIONS:
            with self.subTest(orientation=orientation):
                self.assert_images(self.route(orientation), self.front)

    def test_both_packages_for_all_six_orientations(self):
        for orientation in ORIENTATIONS:
            with self.subTest(orientation=orientation):
                expected = self.front if orientation == "front" else self.back if orientation == "back" else self.front+self.back
                self.assert_images(self.route(orientation, back=True), expected)

    def test_optional_back_none_and_empty_batch_use_front_package(self):
        for missing in (None, np.empty((0, 2, 3, 3), np.float32)):
            self.assert_images(self.route("back", back=True, back_full=missing), self.front)

    def test_back_details_without_back_full_are_never_injected(self):
        out = self.route("back", back_detail_1=self.back[1])
        self.assert_images(out, self.front)
        self.assertTrue(all(row["package"] == "front" for row in json.loads(out[11])["references"]))

    def test_missing_details_compact_picture_ordinals_without_changing_roles(self):
        out = self.route("mixed", back=True, front_detail_1=None, back_detail_2=None)
        self.assert_images(out, [self.front[0],self.front[2],self.front[3],
                                 self.back[0],self.back[1],self.back[3]])
        self.assertEqual([x["input"] for x in json.loads(out[11])["references"]],
                         ["front_full","front_detail_2","front_detail_3",
                          "back_full","back_detail_1","back_detail_3"])

    def test_policy_false_dominates_orientation_and_even_missing_front(self):
        for orientation in ORIENTATIONS:
            out = self.route(orientation, inject=False, back=True, front_full=None)
            self.assert_images(out, [])
            self.assertEqual(out[8], "none")
            self.assertEqual(out[10], "")

    def test_empty_and_secondary_only_use_real_policy_before_router(self):
        for secondary in ("none", "present", "uncertain"):
            rows = scene_rows("back", mode="absent", secondary=secondary)
            policy = self.mod.H3CShotPolicy().run(rows, "", 1, 1, 999,
                                                 self.front[0], self.front[1])
            self.assertFalse(policy[5])
            self.assert_images(self.route(inject=policy[5], back=True, rows=rows), [])

    def test_manual_empty_and_present_keep_existing_policy_semantics(self):
        absent = scene_rows("back", mode="absent")
        empty = self.mod.H3CShotPolicy().run(absent, "", 1, 1, 999,
                                           self.front[0], self.front[1],
                                           manual_force_empty_indices="1")
        self.assert_images(self.route(inject=empty[5], back=True, rows=absent), [])
        present = self.mod.H3CShotPolicy().run(absent, "1", 1, 1, 999,
                                             self.front[0], self.front[1],
                                             manual_force_present_indices="1")
        self.assertTrue(present[5])
        self.assert_images(self.route(inject=present[5], rows=absent), self.front)
        self.assert_images(self.route(inject=present[5], back=True, rows=absent),
                           self.front+self.back)
        with self.assertRaises(ValueError):
            self.mod.H3CShotPolicy().run(absent, "", 1, 1, 999,
                                        self.front[0], self.front[1],
                                        manual_force_present_indices="1",
                                        manual_force_empty_indices="1")

    def test_legacy_plain_and_jsonl_records_conservatively_select_both(self):
        for rows in ("Scene: courtyard", json.dumps({"shot":1,"prompt":"room","subject_mode":"present"})):
            out = self.route(back=True, rows=rows)
            self.assertEqual(out[9], "uncertain")
            self.assert_images(out, self.front+self.back)

    def test_invalid_orientation_is_conservative_and_never_leaks_into_prompt(self):
        for value in ("", "left", None, 3, ["front"]):
            raw = {"subject_mode":"present","secondary_human_presence":"none",
                   "scene":"courtyard","orientation":value}
            record = self.c._normalise_scene_record(json.dumps(raw), 1)
            self.assertEqual(record["orientation"], "uncertain")
            self.assertEqual(record["prompt"], "Scene: courtyard")
            self.assertNotIn("orientation", record["scene_fields"])

    def test_valid_orientation_metadata_survives_round_trip_without_prompt_changes(self):
        for orientation in ORIENTATIONS:
            raw = {"subject_mode":"present","secondary_human_presence":"none",
                   "scene":"courtyard","orientation":orientation,
                   "orientation_reason":"across sampled primary torso views",
                   "orientation_evidence": json.loads(scene_rows(orientation))["orientation_evidence"]}
            record = self.c._normalise_scene_record(json.dumps(raw), 1)
            self.assertEqual(record["orientation"], orientation)
            parsed = self.c._parse_scene_prompts(json.dumps({"shot":1, **record}))[1]
            self.assertEqual(parsed["orientation"], orientation)
            self.assertEqual(parsed["orientation_reason"], raw["orientation_reason"])
            self.assertEqual(record["prompt"], "Scene: courtyard")

    def test_absent_protagonist_has_no_trusted_orientation(self):
        for orientation in ("front", "back"):
            record = self.c._normalise_scene_record(json.dumps({
                "subject_mode":"absent","secondary_human_presence":"present",
                "scene":"courtyard","orientation":orientation}), 1)
            self.assertEqual(record["orientation"], "uncertain")

    def test_router_uses_requested_shot_and_does_not_keep_cross_shot_state(self):
        rows = "\n".join(scene_rows(o, shot=i) for i,o in enumerate(ORIENTATIONS,1))
        cls = self.pkg.NODE_CLASS_MAPPINGS.get("H3CShotReferenceRouter")
        self.assertIsNotNone(cls)
        router = cls()
        for number, orientation in enumerate(ORIENTATIONS,1):
            out = self.route(back=True, rows=rows, shot_number=number, router=router)
            self.assertEqual(out[9], orientation)
            self.assertEqual(json.loads(out[11])["shot"], number)
            expected = self.front if orientation == "front" else self.back if orientation == "back" else self.front+self.back
            self.assert_images(out, expected)

    def test_all_optional_detail_combinations_preserve_dense_slots_and_package_identity(self):
        names = ("front_detail_1", "front_detail_2", "front_detail_3",
                 "back_detail_1", "back_detail_2", "back_detail_3")
        for mask in range(64):
            missing = {name: None for i, name in enumerate(names) if mask & (1 << i)}
            expected = [self.front[0]] + [self.front[i+1] for i in range(3) if names[i] not in missing]
            expected += [self.back[0]] + [self.back[i+1] for i in range(3) if names[i+3] not in missing]
            with self.subTest(mask=mask):
                self.assert_images(self.route("mixed", back=True, **missing), expected)

    def test_bad_index_records_or_enabled_missing_full_fail_fast(self):
        for index in (0,-1,True,1.5,2):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.route(shot_number=index)
        for rows in ("", "{}", scene_rows(shot=2)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.route(shot_prompts=rows)
        with self.assertRaises(ValueError):
            self.route(front_full=None)
        for bad in (np.zeros((1,2,3)), np.zeros((1,0,3,3)), np.zeros((2,2,3,3)), 42):
            with self.assertRaises(ValueError):
                self.route(front_full=bad)

    def test_boolean_gate_must_be_real_boolean(self):
        for value in ("false", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.route(inject=value)

    def test_router_schema_and_registered_display_names(self):
        out = self.route(back=True)
        cls = self.pkg.NODE_CLASS_MAPPINGS["H3CShotReferenceRouter"]
        self.assertEqual(len(out), len(cls.RETURN_TYPES))
        self.assertEqual(cls.RETURN_TYPES[:8], ("IMAGE",)*8)
        self.assertEqual(cls.RETURN_NAMES[:8], tuple(f"ref_image_{i}" for i in range(8)))
        self.assertIn("H3CShotReferenceRouter", self.pkg.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(set(cls.INPUT_TYPES()["optional"]), {
            "front_detail_1","front_detail_2","front_detail_3","back_full",
            "back_detail_1","back_detail_2","back_detail_3"})
        for klass in self.pkg.NODE_CLASS_MAPPINGS.values():
            self.assertEqual(len(klass.RETURN_TYPES), len(klass.RETURN_NAMES))
            self.assertTrue(callable(getattr(klass(), klass.FUNCTION)))

    def test_vlm_appends_orientation_without_second_inference_or_slot_shift(self):
        self.assertEqual(self.c.H3ShotSceneVLM.RETURN_NAMES[:10], OLD_VLM_OUTPUTS)
        calls = []
        class Qwen:
            FUNCTION = "run"
            instances = 0
            def __init__(self):
                type(self).instances += 1
            def run(self, video, custom_prompt, seed, frame_count, image):
                self_outer.assertIsNone(image)
                self_outer.assertIn('"orientation"', custom_prompt)
                self_outer.assertIn("primary protagonist", custom_prompt)
                calls.append((video[:,0,0,0].tolist(),seed,frame_count))
                orientation = ORIENTATIONS[len(calls)-1]
                return (json.dumps({"subject_mode":"present","secondary_human_presence":"none",
                                    "scene":"courtyard","orientation":orientation,
                                    "orientation_evidence": json.loads(scene_rows(orientation))["orientation_evidence"]}),)
        self_outer = self
        self.c.nodes.NODE_CLASS_MAPPINGS["AILab_QwenVL_Advanced"] = Qwen
        video = np.arange(12,dtype=np.float32).reshape(12,1,1,1)
        with patch.object(self.c,"_detect_hard_cuts",return_value=[(i,i+2) for i in range(0,12,2)]):
            out = self.c.H3ShotSceneVLM().run(video,"Qwen", "None (FP16)","auto",
                False,"cpu",1024,32,"auto",1,.18,1)
        self.assertEqual(len(out), 11)
        self.assertEqual(Qwen.instances,1)
        self.assertEqual(calls, [(list(map(float,range(2*i,2*i+2))),i+2,2) for i in range(6)])
        self.assertEqual([json.loads(x)["orientation"] for x in out[0].splitlines()], list(ORIENTATIONS))
        self.assertEqual([json.loads(x)["value"] for x in out[10].splitlines()], list(ORIENTATIONS))
        self.assertEqual(out[0],out[4])
        self.assertEqual(out[1],"")
        self.assertTrue(all("orientation" not in json.loads(x)["scene_fields"] for x in out[5].splitlines()))
        for index in (6,7,8,9):
            self.assertTrue(out[index])

    def test_optional_picture3_behavior_is_unchanged(self):
        upload = self.pkg.NODE_CLASS_MAPPINGS["H3OptionalPicture3"]
        self.assertEqual(upload().load(""), (None,))
        self.assertTrue(upload.VALIDATE_INPUTS(""))
        self.assertEqual(upload.RETURN_NAMES, ("image",))

    def test_single_package_contract_does_not_claim_both_packages(self):
        for orientation, back in (("front", False), ("back", False), ("front", True), ("back", True)):
            contract = self.route(orientation, back=back)[10]
            self.assertNotIn("Both packages", contract)
            self.assertIn("Selected references", contract)
        self.assertIn("Front and Back packages", self.route("turning", back=True)[10])

    def test_front_back_require_consistent_temporal_evidence(self):
        for orientation in ("front", "back"):
            valid = {"begin": orientation, "middle": orientation, "end": orientation,
                     "all_sampled_frames_same": True}
            bad_cases = (None, {}, [], {**valid, "end": "side"},
                         {**valid, "middle": "turning"}, {**valid, "all_sampled_frames_same": False},
                         {**valid, "all_sampled_frames_same": "true"}, {**valid, "begin": [orientation]})
            for evidence in bad_cases:
                with self.subTest(orientation=orientation, evidence=evidence):
                    raw = {"subject_mode": "present", "scene": "room", "orientation": orientation,
                           "orientation_evidence": evidence}
                    record = self.c._normalise_scene_record(json.dumps(raw), 1)
                    self.assertEqual(record["orientation"], "uncertain")
                    rows = json.dumps({"shot": 1, **raw, "prompt": "Scene: room"})
                    self.assert_images(self.route(back=True, rows=rows), self.front + self.back)
            raw = {"subject_mode": "present", "scene": "room", "orientation": orientation,
                   "orientation_evidence": valid}
            self.assertEqual(self.c._normalise_scene_record(json.dumps(raw), 1)["orientation"], orientation)

    def test_sparse_vlm_coverage_cannot_authorize_single_package(self):
        class Qwen:
            FUNCTION = "run"
            def run(self, video, custom_prompt, seed, frame_count, image):
                return (json.dumps({"subject_mode": "present", "scene": "room", "orientation": "back",
                                   "orientation_evidence": json.loads(scene_rows("back"))["orientation_evidence"]}),)
        self.c.nodes.NODE_CLASS_MAPPINGS["AILab_QwenVL_Advanced"] = Qwen
        for total, requested, expected in ((5, 1, "uncertain"), (5, 2, "uncertain"),
                                            (5, 3, "back"), (1, 1, "back"), (2, 2, "back")):
            video = np.zeros((total, 1, 1, 3), np.float32)
            with patch.object(self.c, "_detect_hard_cuts", return_value=[(0, total)]):
                out = self.c.H3ShotSceneVLM().run(video, "Qwen", "None (FP16)", "auto", False,
                                                 "cpu", 1024, requested, "auto", 1, .18, 1)
            record = json.loads(out[0])
            self.assertEqual(record["orientation"], expected)
            self.assertEqual(record["orientation_sampled_frames"], min(total, requested))

    def test_dynamic_shot_counts_use_real_detector_and_whole_pipeline_contract(self):
        torch = sys.modules["torch"]
        torch.cat = lambda chunks, dim=0: np.concatenate(chunks, axis=dim).view(CPUImage)
        torch.argmax = np.argmax
        torch.empty = lambda shape, **kwargs: np.empty(shape, dtype=np.float32).view(CPUImage)
        adapter = sys.modules["_h3_routing_test.h3_hardcut_ref_adapters"]
        class Qwen:
            FUNCTION = "run"
            def __init__(self):
                self.calls = 0
            def run(self, video, custom_prompt, seed, frame_count, image):
                self.calls += 1
                orientation = ORIENTATIONS[(self.calls - 1) % 6]
                return (json.dumps({"subject_mode": "present", "scene": "room", "orientation": orientation,
                    "orientation_evidence": json.loads(scene_rows(orientation))["orientation_evidence"]}),)
        self.c.nodes.NODE_CLASS_MAPPINGS["AILab_QwenVL_Advanced"] = Qwen
        for count in (1, 3, 5, 9, 12, 17, 64):
            with self.subTest(count=count):
                frame = np.arange(count * 4, dtype=np.float32)
                video = (((frame // 4) % 2) + frame / 10000).reshape(-1, 1, 1, 1)
                video = np.repeat(video, 3, axis=3).view(CPUImage)
                depth = (video + .25).view(CPUImage)
                manifest, detected, total, _ = self.mod.H3CShotPlanner().run(video, depth, 999, .5, 1)
                self.assertEqual((detected, total), (count, count * 4))
                self.assertEqual([(s["start"], s["end"]) for s in json.loads(manifest)["shots"]],
                                 [(i * 4, i * 4 + 4) for i in range(count)])
                vlm = self.c.H3ShotSceneVLM().run(video, "Qwen", "None (FP16)", "auto", False,
                                                  "cpu", 1024, 32, "auto", 1, .5, 1)
                self.assertEqual(len(vlm[0].splitlines()), count)
                state = None
                router = self.pkg.NODE_CLASS_MAPPINGS["H3CShotReferenceRouter"]()
                for index in range(count):
                    data = adapter.H3CShotDualRefPad().run(video, depth, manifest, index)
                    rgb_ref, depth_ref = data[1:3]
                    self.assertTrue(np.array_equal(rgb_ref[:4], video[4*index:4*index+4]))
                    self.assertTrue(np.array_equal(depth_ref[:4], depth[4*index:4*index+4]))
                    self.assertTrue(np.all(rgb_ref[4:] == video[4*index+3]))
                    self.assertTrue(np.all(depth_ref[4:] == depth[4*index+3]))
                    policy = self.mod.H3CShotPolicy().run(vlm[0], vlm[1], index+1, count,
                                                         data[5], self.front[0], self.front[1])
                    orientation = ORIENTATIONS[index % 6]
                    routed = self.route(back=True, rows=vlm[0], shot_number=index+1,
                                        inject=policy[5], router=router)
                    expected = self.front if orientation == "front" else self.back if orientation == "back" else self.front+self.back
                    self.assert_images(routed, expected)
                    trimmed, _, report = self.mod.H3CShotTrim().run(rgb_ref, 4, data[4], index+1)
                    state = self.mod.H3COrderedMergeStep().run(state, trimmed, index+1, total,
                        data[7], policy[10], "prompt", report)[0]
                merged, fps, frames, _ = self.mod.H3COrderedMergeFinish().run(state, 24, total)
                self.assertEqual((frames, fps), (total, 24))
                self.assertTrue(np.array_equal(merged, video))

    def test_routed_contract_is_safe_only_after_legacy_picture_cleanup(self):
        rows = scene_rows("front")
        contract = self.route()[10]
        compiler = self.mod.H3CPromptCompiler()
        neutral = "Keep all target references consistent. Preserve the source camera."
        first = compiler.run(rows, 1, neutral, neutral, False, False, False)[0]
        combined = first + "\n" + contract
        for i in range(1, 5):
            self.assertIn(f"<Picture {i}>", combined)
        wrong_order = compiler.run(rows, 1, neutral + "\n" + contract, neutral,
                                   False, False, False)[0]
        self.assertNotIn("<Picture 3>", wrong_order)
        old_roles = compiler.run(rows, 1, "<Picture 2> defines only the face.", neutral,
                                 False, False, False)[0] + "\n" + contract
        self.assertIn("<Picture 2> defines only the face", old_roles)
        self.assertIn("<Picture 2> is the front detail_1", old_roles)
        self.assertNotIn("defines only the face", combined)

    def builder(self):
        spec = importlib.util.spec_from_file_location("_routing_audit_builder", PLUGIN / "tools/build_h3_modular_c.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(getattr(module, "_ensure_scene_vlm_outputs", None)),
                        "builder must synchronize VLM outputs without running a workflow build")
        return module

    def test_builder_output_extension_is_idempotent_without_json_generation(self):
        builder = self.builder()
        names = (*OLD_VLM_OUTPUTS, "shot_orientation")
        for count in (3, 10, 11):
            node = {"outputs": [{"name": n, "type": "STRING", "slot_index": i, "links": [i+100]}
                                for i, n in enumerate(names[:count])]}
            old = copy.deepcopy(node["outputs"])
            builder._ensure_scene_vlm_outputs(node)
            self.assertEqual(node["outputs"][:count], old)
            self.assertEqual([x["name"] for x in node["outputs"]], list(names))
            result = copy.deepcopy(node)
            builder._ensure_scene_vlm_outputs(node)
            self.assertEqual(node, result)

    def test_builder_rejects_changed_old_output_order(self):
        builder = self.builder()
        for outputs in ([{"name": "report", "type": "STRING"}],
                        [{"name": "shot_prompts", "type": "IMAGE"}]):
            with self.assertRaises(ValueError):
                builder._ensure_scene_vlm_outputs({"outputs": outputs})

    def test_both_builder_paths_statically_sync_vlm_schema(self):
        # Full builder execution is forbidden because it writes workflow JSON.
        tree = ast.parse((PLUGIN / "tools/build_h3_modular_c.py").read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ("build", "build_a2_force_refs"):
            calls = [node.func.id for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            self.assertIn("_ensure_scene_vlm_outputs", calls, name)

    def test_modular_test_fixture_restores_host_imports(self):
        spec = importlib.util.spec_from_file_location("_modular_fixture_audit", PLUGIN / "tests/test_h3_modular_c.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = dict(sys.modules)
        result = unittest.TestResult()
        suite = unittest.TestSuite([module.ModularCContractTests("test_registered_nodes_have_comfy_contracts")])
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors)
        self.assertIs(sys.modules.get("torch"), before.get("torch"))
        self.assertIs(sys.modules.get("nodes"), before.get("nodes"))
        self.assertEqual({k for k in sys.modules if k.startswith("h3_modular_test_pkg")},
                         {k for k in before if k.startswith("h3_modular_test_pkg")})

    def test_selected_bad_detail_is_not_silently_dropped(self):
        for bad in (42, np.zeros((1, 2, 3)), np.zeros((2, 2, 3, 3)), np.zeros((1, 2, 3, 1))):
            with self.subTest(shape=getattr(bad, "shape", None)), self.assertRaises(ValueError):
                self.route(front_detail_2=bad)


if __name__ == "__main__":
    unittest.main()
