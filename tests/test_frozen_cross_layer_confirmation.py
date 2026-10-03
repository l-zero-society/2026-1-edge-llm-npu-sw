import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_frozen_cross_layer_confirmation as confirm


class FrozenCrossLayerConfirmationTest(unittest.TestCase):
    def test_selected_configuration_requires_exact_changed_layers(self):
        self.assertEqual(confirm.CHANGED_LAYERS, {13, 16, 17})
        document = json.loads(confirm.SELECTED_PATH.read_text())
        changed = {v["layer"] for v in document["layers"].values() if v["source"] != "old"}
        self.assertEqual(document["changed_layers"], 3)
        self.assertEqual(changed, {13, 16, 17})

    def test_stable_id_exclusion(self):
        excluded = {"tree_id:t1", "anchor_id:a1"}
        self.assertTrue(confirm.is_previously_used({"tree_id": "t1", "index": 9}, excluded))
        self.assertTrue(confirm.is_previously_used({"anchor_id": "a1", "index": 2}, excluded))
        self.assertFalse(confirm.is_previously_used({"tree_id": "fresh", "index": 9}, excluded))

    def test_source_index_is_not_globally_unique(self):
        excluded = {"tree_id:validation-tree"}
        calibration = {"tree_id": "calibration-tree", "index": 4}
        validation = {"tree_id": "validation-tree", "index": 4}
        self.assertFalse(confirm.is_previously_used(calibration, excluded))
        self.assertTrue(confirm.is_previously_used(validation, excluded))

    def test_deterministic_unseen_order(self):
        examples = [{"stable_id": f"id-{i}"} for i in range(40)]
        first = [confirm.stable_id(x) for x in confirm.deterministic_unseen_order(examples)]
        second = [confirm.stable_id(x) for x in confirm.deterministic_unseen_order(list(reversed(examples)))]
        self.assertEqual(first, second)

    def test_local_dataset_size_policy(self):
        chosen, fallback = confirm.local_dataset_policy(list(range(40)))
        self.assertEqual(len(chosen), 32)
        self.assertFalse(fallback)
        chosen, fallback = confirm.local_dataset_policy(list(range(24)))
        self.assertEqual(len(chosen), 24)
        self.assertFalse(fallback)
        self.assertEqual(len(chosen) * 32, 768)

    def test_fallback_below_24(self):
        chosen, fallback = confirm.local_dataset_policy(list(range(23)))
        self.assertEqual(chosen, [])
        self.assertTrue(fallback)

    def test_wikitext_windows_are_deterministic_and_nonoverlapping(self):
        tokens = list(range(32 * 160))
        a = confirm.wikitext_windows(tokens)
        b = confirm.wikitext_windows(tokens)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)
        self.assertEqual(sum(len(x["targets"]) for x in a), 1024)
        self.assertEqual(a[0]["window_end"], a[1]["window_start"])

    def test_no_previous_data_overlap(self):
        excluded = {"tree_id:used"}
        pool = [{"stable_id": "fresh", "tree_id": "fresh", "targets": list(range(32))}]
        self.assertFalse(any(confirm.is_previously_used(e, excluded) for e in pool))

    def test_confirmation_declares_no_search_or_calibration(self):
        source = Path(confirm.__file__).read_text()
        self.assertIn("search_performed=False", source)
        self.assertIn("calibration_performed=False", source)
        self.assertNotIn("calibrate_layer_joint(", source)

    def test_safety_gate(self):
        old = dict(kl=1.0, nmse=1.0, top1=.5)
        self.assertTrue(confirm.confirmation_gate(old, dict(kl=1.05, nmse=1.05, top1=.49))["pass"])
        self.assertFalse(confirm.confirmation_gate(old, dict(kl=1.051, nmse=1.0, top1=.5))["pass"])

    def test_strict_improvement(self):
        old = dict(kl=1.0, nmse=1.0, top1=.5)
        self.assertTrue(confirm.strict_improvement(old, dict(kl=.99, nmse=1.0, top1=.5)))
        self.assertFalse(confirm.strict_improvement(old, dict(kl=.99, nmse=1.01, top1=.5)))

    def test_bootstrap_is_deterministic(self):
        def raw(nll, kl, se, top1):
            return dict(metric=dict(n=2, rows=1, se=se, ae=.1, re=2., qe=2., cross=2., cos=1.),
                        n=1, nll=nll, kl=kl, top1=top1, in5=1., overlap=1.)
        records = [dict(scores={"old_absmax": raw(.5, .2, .2, 0.),
                                "frozen_cross_layer_absmax": raw(.4, .1, .1, 1.)}) for _ in range(3)]
        a = confirm.bootstrap(records, replicates=20, seed=4)
        b = confirm.bootstrap(records, replicates=20, seed=4)
        self.assertEqual(a, b)

    def test_checkpoint_fingerprint_validation(self):
        keys = {"FP": "a", "old_absmax": "b", "frozen_cross_layer_absmax": "c"}
        self.assertTrue(confirm.checkpoint_valid({"checkpoint_keys": keys}, keys))
        self.assertFalse(confirm.checkpoint_valid({"checkpoint_keys": dict(keys, FP="x")}, keys))


if __name__ == "__main__":
    unittest.main()
