from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gpalu_pot_fusion_pilot as gpalu


class GpaluPotFusionTests(unittest.TestCase):
    def test_int8_products_fit_int16_and_worst_cases(self):
        a=np.array([-127,-127,127,127],np.int16);b=np.array([-128,127,-128,127],np.int16)
        products=a.astype(np.int32)*b.astype(np.int32)
        np.testing.assert_array_equal(products,[16256,-16129,-16256,16129])
        self.assertTrue(np.all((products>=-32768)&(products<=32767)))

    def test_signed_rne_positive_negative_and_half_even(self):
        values=np.array([1,3,5,7,-1,-3,-5,-7])
        np.testing.assert_array_equal(gpalu.signed_rne_right_shift(values,1),[0,2,2,4,0,-2,-2,-4])

    def test_kg_zero_identity(self):
        x=np.array([-300,-128,-1,0,127,300])
        np.testing.assert_array_equal(gpalu.signed_rne_right_shift(x,0),x)

    def test_kg_range_exact(self):
        self.assertEqual(gpalu.KG_VALUES,tuple(range(8)))
        with self.assertRaises(ValueError):gpalu.signed_rne_right_shift([1],8)

    def test_post_shift_clipping_counts(self):
        q,info=gpalu.gpalu_codes(np.array([127,-127,127]),np.array([127,127,-128]),0)
        np.testing.assert_array_equal(q,[127,-128,-128])
        self.assertEqual(info["positive_clip_count"],1);self.assertEqual(info["negative_clip_count"],2)

    def test_output_scale(self):
        self.assertEqual(gpalu.output_scale(.5,.25,3),1.)

    def test_candidate_search_includes_every_kg(self):
        rows=[self.row(k,1+k) for k in gpalu.KG_VALUES]
        self.assertEqual(gpalu.select_best_kg(rows)["kG"],0)

    def test_best_kg_uses_post_nmse_and_clipping_not_rejection(self):
        rows=[self.row(k,10+k) for k in gpalu.KG_VALUES]
        rows[2].update(post_gpalu_nmse_balanced=.5,post_gpalu_nmse_worst=.5,clip_rate=.2)
        self.assertEqual(gpalu.select_best_kg(rows)["kG"],2)

    def test_fixed_s10_isolation_and_joint_selector(self):
        current=self.s10row(0,1.,2.);better=self.s10row(1,2.,1.)
        selected,old=gpalu.select_joint([current,better],1.)
        self.assertIs(old,current);self.assertIs(selected,better)

    def test_deterministic_tie_break(self):
        rows=[self.row(k,1.) for k in gpalu.KG_VALUES]
        self.assertEqual(gpalu.select_best_kg(rows)["kG"],0)

    def test_e2e_classification_uses_multimetric_pattern(self):
        fp={"kl":1.,"nmse":1.,"ppl":1.,"top1":.8}
        static={"kl":2.,"nmse":2.,"ppl":2.,"top1":.7}
        joint={"kl":.9,"nmse":.9,"ppl":1.5,"top1":.75}
        self.assertEqual(gpalu.classify_e2e(fp,static,joint),"PROCEED_TO_LARGE_BATCH")
        unsupported={"kl":1.5,"nmse":1.5,"ppl":1.5,"top1":.6}
        self.assertEqual(gpalu.classify_e2e(fp,static,unsupported),"STATIC_KG_NOT_SUPPORTED_BY_PILOT")

    def test_down_scale_alignment(self):
        sh=gpalu.output_scale(.1,.2,2);down=.04
        self.assertAlmostEqual(sh/down,2.);self.assertAlmostEqual(np.log2(sh/down),1.)

    def test_candidate_grid_center_and_original(self):
        rows=gpalu.candidate_scales(1.,3.)
        self.assertEqual(sum(r["s10_j"]==0 for r in rows),1);self.assertIn(3.,[r["s10"] for r in rows])

    def test_baseline_immutability_helper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.write_text("a");before=gpalu.sha(p);p.write_text("b");self.assertNotEqual(before,gpalu.sha(p))

    @staticmethod
    def row(k,score):
        return {"kG":k,"post_gpalu_nmse_balanced":score,"post_gpalu_nmse_worst":score,
                "clip_rate":0.,"gpalu_incremental_nmse_balanced":score}

    @staticmethod
    def s10row(j,s10,score):
        return {"s10_j":j,"s10":s10,"post_gpalu_nmse_balanced":score,"post_gpalu_nmse_worst":score,
                "clip_rate":0.,"pre_gpalu_nmse_balanced":score,"gelu_nmse_balanced":score,"gate_nmse_balanced":score}


if __name__=="__main__":unittest.main()
