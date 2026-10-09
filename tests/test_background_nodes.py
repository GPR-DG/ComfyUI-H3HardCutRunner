import sys
import pathlib
import unittest
import importlib.util
import types
import math
import json
import tempfile
import warnings
from unittest.mock import patch

import torch
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import h3_background_nodes as nodes


class BackgroundNodeContractTests(unittest.TestCase):
    def start(self, label="", db_path=":memory:"):
        task, _ = nodes.H3CBackgroundTaskStart().run(label, db_path)
        self.addCleanup(task.cleanup)
        return task

    def scene(self, seed=3):
        rng = np.random.default_rng(seed)
        image = np.full((280, 360, 3), (100, 125, 145), np.uint8)
        for i in range(140):
            x, y = int(rng.integers(10, 350)), int(rng.integers(10, 270))
            color = tuple(int(v) for v in rng.integers(20, 235, 3))
            cv2.circle(image, (x, y), int(rng.integers(2, 8)), color, -1)
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, .55, color, 1)
        return torch.from_numpy(image.copy()).float().unsqueeze(0) / 255

    def candidate(self, task, image=None, number=1, background_only=True, mask=None):
        image = self.scene() if image is None else image
        return nodes.H3CBestBackgroundFrame().run(
            task, image, number, background_only, 320, 5, person_masks=mask)

    def resolve(self, task, image=None, number=1):
        candidate = self.candidate(task, image, number)[0]
        return nodes.H3CBackgroundResolve().run(task, candidate)

    def wash(self, task, view, clean):
        token, claimed, _ = nodes.H3CBackgroundWashClaim().run(task, view)
        self.assertTrue(claimed)
        return nodes.H3CBackgroundAttachClean().run(task, view, clean, wash_token=token)

    def test_all_nodes_are_registered(self):
        expected = {
            "H3CBackgroundTaskStart",
            "H3CBestBackgroundFrame",
            "H3CBackgroundResolve",
            "H3CBackgroundWashClaim",
            "H3CBackgroundAttachClean",
            "H3CBackgroundTaskCleanup",
        }
        self.assertTrue(expected.issubset(nodes.NODE_CLASS_MAPPINGS))
        self.assertTrue(expected.issubset(nodes.NODE_DISPLAY_NAME_MAPPINGS))

    def test_tasks_are_unique_and_isolated(self):
        first, first_id = nodes.H3CBackgroundTaskStart().run("", ":memory:")
        second, second_id = nodes.H3CBackgroundTaskStart().run("", ":memory:")
        self.assertNotEqual(first_id, second_id)
        self.assertNotEqual(first.task_id, second.task_id)
        first.cleanup()
        second.cleanup()
        with self.assertRaises(ValueError):
            first.cache.index()

    def test_best_frame_keeps_background_only_contract(self):
        task, _ = nodes.H3CBackgroundTaskStart().run("", ":memory:")
        try:
            frames = np.zeros((2, 32, 32, 3), dtype=np.float32)
            frames[0, 8:24, 8:24] = 0.25
            frames[1, 8:24, 8:24] = 0.75
            candidate, selected, mask, index, report = nodes.H3CBestBackgroundFrame().run(
                task, frames, 1, True, 32, 2
            )
            self.assertEqual(candidate.task_id, task.task_id)
            self.assertEqual(selected.shape, (1, 32, 32, 3))
            self.assertIsNone(mask)
            self.assertIn("confirmed_background_only", report)
            self.assertIn(index, (0, 1))
        finally:
            task.cleanup()

    def test_start_always_invalidates_cached_execution(self):
        changed = getattr(nodes.H3CBackgroundTaskStart, "IS_CHANGED", None)
        self.assertIsNotNone(changed, "a cached TaskStart would reuse an ended video scope")
        self.assertTrue(math.isnan(changed("", ":memory:")))

    def test_repeated_label_and_database_do_not_resume_other_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(pathlib.Path(directory) / "cache.sqlite3")
            first = self.start("same-label", path)
            second = None
            try:
                self.resolve(first)
                second = self.start("same-label", path)
                self.assertNotEqual(first.task_id, second.task_id)
                self.assertEqual(second.cache.index(), [])
            finally:
                # Close before Windows TemporaryDirectory tries to unlink SQLite.
                if second is not None:
                    second.cleanup()
                first.cleanup()

    def test_real_tensor_image_and_mask_outputs_own_selected_pixels(self):
        task = self.start()
        image = self.scene()
        mask = torch.zeros(image.shape[:3])
        mask[:, 100:120, 150:170] = 1
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            candidate, selected, selected_mask, _, report = self.candidate(
                task, image, background_only=False, mask=mask)
        self.assertIsInstance(selected, torch.Tensor)
        self.assertIsInstance(selected_mask, torch.Tensor)
        self.assertEqual(selected_mask.shape, (1, 280, 360))
        self.assertEqual(selected.dtype, torch.float32)
        self.assertEqual(selected.device.type, "cpu")
        torch.testing.assert_close(selected, image)
        torch.testing.assert_close(selected_mask, mask)
        before = candidate.image.copy()
        selected.zero_()
        np.testing.assert_array_equal(candidate.image, before)
        self.assertFalse(any("not writable" in str(w.message) for w in emitted))
        self.assertEqual(json.loads(report)["raw_shot_frames"], 1)

    def test_boolean_and_integer_contracts_are_not_silently_coerced(self):
        task = self.start()
        image = self.scene()
        with self.assertRaises(ValueError):
            self.candidate(task, image, background_only="False")
        with self.assertRaises(ValueError):
            self.candidate(task, image, number=True)

    def test_background_only_rejects_any_foreground(self):
        task = self.start()
        image = self.scene()
        mask = torch.zeros(image.shape[:3]); mask[:, 0, 0] = .01
        with self.assertRaises(ValueError):
            self.candidate(task, image, background_only=True, mask=mask)

    def test_unknown_foreground_is_not_inferred_as_empty(self):
        task = self.start()
        candidate = self.candidate(task, background_only=False)[0]
        out = nodes.H3CBackgroundResolve().run(task, candidate)
        self.assertFalse(candidate.background_only)
        self.assertEqual(out[1].report["diagnostic_decision"], "UNCERTAIN")
        self.assertEqual(out[4:], ("PENDING", False))
        self.assertIsNone(out[2])

    def test_A_B_A_reuses_same_task_READY_background(self):
        task = self.start(); a = self.scene(); b = self.scene(55)
        va = self.resolve(task, a)[0]; self.wash(task, va, a)
        vb = self.resolve(task, b, 2)[0]; self.wash(task, vb, b)
        out = self.resolve(task, a, 3)
        self.assertEqual(out[1].decision, "REUSE_EXISTING")
        self.assertEqual(out[0].view_id, va.view_id)
        self.assertEqual(out[4:], ("READY", True))
        torch.testing.assert_close(out[2], a)

    def test_large_crop_is_NEW_VIEW_not_wrong_clean_reuse(self):
        task = self.start(); a = self.scene()
        va = self.resolve(task, a)[0]; self.wash(task, va, a)
        out = self.resolve(task, a[:, 40:230, 60:300], 2)
        self.assertEqual(out[1].decision, "NEW_VIEW", out[1].report)
        self.assertEqual(out[0].environment_id, va.environment_id)
        self.assertNotEqual(out[0].view_id, va.view_id)
        self.assertIsNone(out[2])

    def test_next_video_same_pixels_never_reuses_ended_task(self):
        first = self.start(); a = self.scene()
        va = self.resolve(first, a)[0]; self.wash(first, va, a)
        nodes.H3CBackgroundTaskCleanup().run(first, completion="all-consumers-done")
        second = self.start(); out = self.resolve(second, a)
        self.assertEqual(out[1].decision, "NEW_ENVIRONMENT")
        self.assertNotEqual(out[0].view_id, va.view_id)

    def test_automatic_attach_requires_nonempty_claim_token(self):
        task = self.start(); image = self.scene(); view = self.resolve(task, image)[0]
        with self.assertRaisesRegex(ValueError, "token"):
            nodes.H3CBackgroundAttachClean().run(task, view, image)
        self.assertEqual(task.cache.get(view.view_id).state, "PENDING")

    def test_claim_attach_and_stale_completion_are_guarded(self):
        task = self.start(); image = self.scene(); view = self.resolve(task, image)[0]
        claim = nodes.H3CBackgroundWashClaim()
        token, won, _ = claim.run(task, view); self.assertTrue(won)
        self.assertFalse(claim.run(task, view)[1])
        task.cache.release_wash(view.view_id, token)
        replacement, won, _ = claim.run(task, view); self.assertTrue(won)
        with self.assertRaisesRegex(ValueError, "stale"):
            nodes.H3CBackgroundAttachClean().run(task, view, image, wash_token=token)
        out = nodes.H3CBackgroundAttachClean().run(task, view, image, wash_token=replacement)
        self.assertEqual(out[0].state, "READY")
        torch.testing.assert_close(out[1], image)
        with self.assertRaises(ValueError):
            nodes.H3CBackgroundAttachClean().run(task, view, image*.9, wash_token=replacement)

    def test_removal_mask_is_transmitted_not_guessed_from_color(self):
        task = self.start(); image = self.scene()
        candidate = self.candidate(task, image, background_only=False)[0]
        view = nodes.H3CBackgroundResolve().run(task, candidate)[0]
        token = nodes.H3CBackgroundWashClaim().run(task, view)[0]
        mask = torch.zeros(image.shape[:3]); mask[:, 100:155, 150:190] = 1
        out = nodes.H3CBackgroundAttachClean().run(
            task, view, image*.9, removal_mask=mask, wash_token=token)
        np.testing.assert_array_equal(out[0].reuse_exclusion_mask, mask[0].numpy())
        self.assertEqual(out[0].candidate.key, candidate.key)

    def test_cross_task_candidate_and_view_rejected(self):
        first, second = self.start(), self.start()
        candidate = self.candidate(first)[0]
        with self.assertRaises(ValueError):
            nodes.H3CBackgroundResolve().run(second, candidate)
        view = nodes.H3CBackgroundResolve().run(first, candidate)[0]
        with self.assertRaises(ValueError):
            nodes.H3CBackgroundWashClaim().run(second, view)
        with self.assertRaises(ValueError):
            nodes.H3CBackgroundAttachClean().run(second, view, self.scene(), wash_token="invalid")

    def test_cleanup_is_active_sink_with_completion_dependency(self):
        cls = nodes.H3CBackgroundTaskCleanup
        self.assertTrue(getattr(cls, "OUTPUT_NODE", False))
        self.assertIn("completion", cls.INPUT_TYPES()["required"])
        task = self.start(); self.resolve(task)
        out = cls().run(task, completion=None)
        self.assertTrue(out[1]); self.assertTrue(task.cleaned)
        with self.assertRaises(ValueError):
            self.candidate(task)

    def test_every_node_has_matching_schema_and_callable(self):
        for cls in nodes.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(cls.INPUT_TYPES()["required"])
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
            self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))
            self.assertTrue(cls.CATEGORY)

    def load_root(self, suffix):
        root = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "_background_root_"+suffix, root/"__init__.py", submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        return package

    def test_real_torch_root_import_preserves_old_and_new_registrations(self):
        host = types.ModuleType("nodes"); host.NODE_CLASS_MAPPINGS = {}
        with patch.dict(sys.modules, {"nodes":host}):
            package = self.load_root("normal")
            self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 25)
            for name in ("H3CShotSelectPad", "H3CShotDualRefPad", "H3CShotGuideWindowPlanner",
                         "H3CShotGuideWindowSelect", "H3CShotGuideWindowAppend", "H3CShotReferenceRouter"):
                self.assertIn(name, package.NODE_CLASS_MAPPINGS)
            self.assertTrue(set(nodes.NODE_CLASS_MAPPINGS) <= set(package.NODE_CLASS_MAPPINGS))

    def test_background_import_failure_keeps_old_nodes_and_logs_traceback(self):
        import builtins
        original = builtins.__import__
        host = types.ModuleType("nodes"); host.NODE_CLASS_MAPPINGS = {}
        for error in (ImportError("missing cv2"), SyntaxError("broken background module")):
            def importing(name, *args, **kwargs):
                if name == "h3_background_nodes":
                    raise error
                return original(name, *args, **kwargs)
            with patch.dict(sys.modules, {"nodes":host}), patch("builtins.__import__", importing):
                with self.assertLogs(level="WARNING") as logged:
                    package = self.load_root(type(error).__name__)
                self.assertIn("H3CShotReferenceRouter", package.NODE_CLASS_MAPPINGS)
                self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 19)
                self.assertTrue(any("Traceback" in s for s in logged.output))


if __name__ == "__main__":
    unittest.main()

