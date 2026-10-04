import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_mixed_linear_gelu_calibration as mixed


class MixedLinearGeluTests(unittest.TestCase):
    def test_classification_and_counts(self):
        inventory=mixed.expected_inventory()
        self.assertEqual(len(inventory),126)
        classes=[mixed.classify_op(n) for n in inventory]
        self.assertEqual(classes.count("ORDINARY_INT8"),108)
        self.assertEqual(classes.count("GATE_PREACT_INT10"),18)

    def test_output_clamps(self):
        self.assertEqual(mixed.output_range("ORDINARY_INT8"),(-128,127))
        self.assertEqual(mixed.output_range("GATE_PREACT_INT10"),(-512,511))

    def test_q10_address(self):
        np.testing.assert_array_equal(mixed.gelu_base.address([-512,0,511]),[0,512,1023])

    def test_effective_shift_feasibility(self):
        self.assertEqual(mixed.effective_bounds(np.array([10,20])),(-8,7))

    def test_absmax_selector(self):
        q,k,info=mixed.selector.select_absmax(np.array([[0.,1.]]),.01,np.array([10]))
        self.assertEqual(q.shape,(1,2));self.assertTrue(0<=info["effective_shift_min"]<=31)

    def test_balanced_nmse(self):
        self.assertEqual(mixed.balanced([1.,3.]),2.)
        self.assertAlmostEqual(mixed.nmse(np.array([2.]),np.array([1.])),1.)

    def test_deterministic_tie_break(self):
        a=dict(score=1,worst_nmse=1,output_saturation_rate=0,input_clip_rate=0,sx_base=1,output_scale=1)
        b=dict(a,sx_base=2)
        self.assertLess(mixed.candidate_key(a,1,1),mixed.candidate_key(b,1,1))

    def test_gelu_identity_and_table(self):
        x=np.array([-1.,0.,1.]);np.testing.assert_array_equal(mixed.gelu_base.gelu(x),mixed.gelu_base.gelu(x))
        table=mixed.gelu_base.make_lut(.1,.1);self.assertEqual(table.shape,(1024,));self.assertEqual(len(table.tobytes()),1024)

    def test_percentile_100_absmax(self):
        values=np.arange(1,101.)
        candidates=mixed.gelu_base.percentile_candidates(values)
        self.assertEqual(candidates[-1][0],100.);self.assertAlmostEqual(candidates[-1][1],100/127)

    def test_artifact_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json";value={"status":"MIXED_LINEAR_GELU_BASELINE"};p.write_text(json.dumps(value))
            self.assertEqual(json.loads(p.read_text()),value)

    def test_checkpoint_fingerprint_rejection(self):
        self.assertNotEqual(mixed.digest({"a":1}),mixed.digest({"a":2}))


if __name__=="__main__":unittest.main()
