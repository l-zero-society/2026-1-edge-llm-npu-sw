import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_cached_decode_ptq as c


class CachedDecodePTQMathTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(17)
        groups = {"prefill": rng.normal(size=(7, 11)), "decode": rng.normal(size=(5, 11))}
        shift = np.full(6, 12, np.uint8)
        return groups, shift

    def test_reservoir_deterministic_and_bounded(self):
        rows = np.arange(40 * 3).reshape(40, 3)
        result = []
        for _ in range(2):
            sampler = c.Reservoir(8, 9)
            for i, value in enumerate(rows): sampler.add(value, 2, "prefill", 4, i)
            result.append(sampler.arrays())
        np.testing.assert_array_equal(result[0][0], result[1][0])
        self.assertEqual(len(result[0][0]), 8)
        self.assertEqual(result[0][1], result[1][1])

    def test_sx_search_is_activation_only(self):
        groups, shift = self.fixture()
        with patch.object(c.base, "integer_dot", side_effect=AssertionError("GEMM in sX search")):
            best, table = c.search_sx(groups, .03, shift)
        self.assertEqual(len(table), 9); self.assertIn(best, table)
        self.assertTrue(np.isfinite(best["score"]))

    def test_sx_tie_break_prefers_old(self):
        groups = {"prefill": np.zeros((2, 5)), "decode": np.zeros((3, 5))}
        best, _ = c.search_sx(groups, .125, np.full(2, 10, np.uint8))
        self.assertEqual(best["j"], 0); self.assertEqual(best["sx"], .125)

    def test_s10_reuses_supplied_acc(self):
        rng=np.random.default_rng(3);data=dict(acc=rng.integers(-300,300,size=(4,6),dtype=np.int64),
            ref=rng.normal(size=(4,6)),ks=np.array([-1,0,1,0]),labels=np.array(["prefill","prefill","decode","decode"]))
        sw=np.linspace(.001,.004,6)
        with patch.object(c.base,"integer_dot",side_effect=AssertionError("repeated ACC")):
            best,table=c.search_s10(data,.02,.03,sw)
        self.assertLessEqual(len(table),17);self.assertIn(best,table)

    def test_effective_shift_feasibility(self):
        rng=np.random.default_rng(5);data=dict(acc=rng.integers(-100,100,size=(3,4),dtype=np.int64),
            ref=np.ones((3,4)),ks=np.array([7,0,-8]),labels=np.array(["prefill","decode","decode"]))
        result=c.evaluate_s10(data,.01,.02,np.ones(4)*.001)
        if result is not None:
            self.assertGreaterEqual(result["effective_shift_min"],0)
            self.assertLessEqual(result["effective_shift_max"],31)

    def test_runtime_selector_is_existing_function(self):
        groups,shift=self.fixture();x=np.concatenate(list(groups.values()))
        a=c.choose_rows(x,.03,shift);b=c.row.select_rows(x,.03,shift)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_response_template_source_constant(self):
        protocol=json_load(c.ROOT/"diagnostics/response_quant_eval/protocol.json")
        self.assertEqual(protocol["template_source"],"local GGUF tokenizer.chat_template")

    def test_no_production_output_path(self):
        self.assertTrue(str(c.OUT).endswith("diagnostics/cached_decode_ptq"))
        self.assertTrue(str(c.OUT/"proposed").endswith("diagnostics/cached_decode_ptq/proposed"))

    def test_mixed_quantization_keeps_batch_lanes_separate(self):
        rng=np.random.default_rng(31);x=rng.normal(size=(3,4,7)).astype(np.float32)
        wq,sw=c.base.quantize_weight(rng.normal(size=(5,7)));sx=.03;s10=.02
        spec=dict(wq=wq,sx=sx,s10=s10,k=0,params=c.Profile().approximate(sx*sw/s10))
        name="model.layers.0.mlp.down_proj"
        import torch
        class Linear:
            def forward(self,value):return torch.from_numpy(value.numpy().astype(np.float64)@rng.normal(size=(5,7)).T).float()
        module=Linear();source=type("S",(),dict(modules={name:module},torch=torch))()
        original=module.forward
        modes={"FP":None,"old_general":{name:spec},"calibrated":{name:spec}}
        with c.patch_mixed_down(source,modes,"decode"):
            result=module.forward(torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(result[1:2],c.quantized_one(x[1:2],spec))
        np.testing.assert_array_equal(result[2:3],c.quantized_one(x[2:3],spec))
        self.assertIs(module.forward.__func__,original.__func__)


def json_load(path):
    import json
    return json.loads(path.read_text())


if __name__ == "__main__": unittest.main()
