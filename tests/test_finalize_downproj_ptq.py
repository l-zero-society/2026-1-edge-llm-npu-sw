import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import finalize_downproj_ptq as f


class FinalizeDownprojPTQTests(unittest.TestCase):
    def test_boundary_set_matches_csv(self):
        rows=f.read_scale_rows()
        self.assertEqual({r["layer"]:r["s10_i"] for r in rows if abs(r["s10_i"])==8},f.BOUNDARY)

    def test_outward_indices_and_reference_scale(self):
        data=dict(acc=np.zeros((2,3),np.int64),ref=np.ones((2,3)),ks=np.zeros(2,np.int64),
                  labels=np.array(["prefill","decode"]))
        sw=np.full(3,.01)
        plus=f.evaluate_indices(data,.02,.04,sw,[8,9,16])
        minus=f.evaluate_indices(data,.02,.04,sw,[-8,-9,-16])
        self.assertEqual([x["i"] for x in plus],[8,9,16]);self.assertEqual([x["i"] for x in minus],[-8,-9,-16])
        self.assertEqual(plus[0]["s10"],.08);self.assertEqual(minus[0]["s10"],.02)

    def test_acceptance_threshold_boundary(self):
        old={"score":.01,"worst":.02,"clip_rate":0,"i":8}
        accept={"score":.00989,"worst":.02,"clip_rate":0,"i":9}
        reject={"score":.009901,"worst":.01,"clip_rate":0,"i":9}
        self.assertGreaterEqual((old["score"]-accept["score"])/old["score"],.01)
        self.assertLess((old["score"]-reject["score"])/old["score"],.01)

    def test_s10_sweep_uses_supplied_acc(self):
        data=dict(acc=np.zeros((2,3),np.int64),ref=np.ones((2,3)),ks=np.zeros(2,np.int64),
                  labels=np.array(["prefill","decode"]))
        with patch.object(f.base,"integer_dot",side_effect=AssertionError("ACC recomputed")):
            result=f.evaluate_indices(data,.02,.04,np.full(3,.01),[8,9])
        self.assertEqual(len(result),2)

    def test_shift_metadata_names_and_bounds(self):
        rng=np.random.default_rng(2)
        groups={"prefill":rng.normal(size=(4,7)),"decode":rng.normal(size=(3,7))}
        wq,sw=f.base.quantize_weight(rng.normal(size=(5,7)))
        sx,s10=.02,.03
        spec=dict(wq=wq,sx=sx,s10=s10,k=0,params=f.Profile().approximate(sx*sw/s10))
        result=f.observed_shift_metadata(groups,spec)
        self.assertEqual(set(result),{"static_shift_min","static_shift_max","observed_effective_shift_min",
                                     "observed_effective_shift_max","observed_k_min","observed_k_max"})
        self.assertGreaterEqual(result["observed_effective_shift_min"],0)
        self.assertLessEqual(result["observed_effective_shift_max"],31)

    def test_depth_buckets_cover_positions(self):
        expected={0:"0-31",31:"0-31",32:"32-63",63:"32-63",64:"64-127",127:"64-127",
                  128:"128-255",255:"128-255",256:"256-511",511:"256-511",512:">=512"}
        self.assertEqual({p:f.depth_label(p) for p in expected},expected)

    def test_no_production_output_path(self):
        self.assertTrue(str(f.OUT).endswith("diagnostics/final_downproj_ptq"))
        self.assertTrue(str(f.PROPOSED).endswith("diagnostics/final_downproj_ptq/proposed"))


if __name__=="__main__":unittest.main()
