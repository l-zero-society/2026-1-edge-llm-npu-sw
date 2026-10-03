import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "calibration/frozen_linear_quant_v1/parameters.json"
MANIFEST = ROOT / "calibration/frozen_linear_quant_v1/manifest.json"
SOURCE = ROOT / "diagnostics/absmax_cross_layer_calibration/selected_configuration.json"
OLD = ROOT / "diagnostics/final_downproj_ptq/decisions.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenLinearQuantV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = json.loads(CANONICAL.read_text())
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.source = json.loads(SOURCE.read_text())
        cls.old = json.loads(OLD.read_text())

    def test_artifacts_exist(self):
        self.assertTrue(CANONICAL.is_file())
        self.assertTrue(MANIFEST.is_file())

    def test_exact_expected_module_set(self):
        expected = {f"model.layers.{i}.mlp.down_proj" for i in range(18)}
        self.assertEqual(set(self.parameters["layers"]), expected)
        self.assertEqual(len(self.parameters["layers"]), 18)

    def test_canonical_content_matches_selected_configuration(self):
        self.assertEqual(self.parameters, self.source)

    def test_only_expected_layers_differ_from_historical_reference(self):
        changed = []
        for name, selected in self.parameters["layers"].items():
            old = self.old["final"][str(selected["layer"])]
            if selected["sx"] != old["sx_base"] or selected["s10"] != old["s10"]:
                changed.append(selected["layer"])
            else:
                self.assertEqual(selected["source"], "old")
        self.assertEqual(sorted(changed), [13, 16, 17])

    def test_manifest_hashes(self):
        self.assertEqual(self.manifest["source_selected_configuration_sha256"], sha256(SOURCE))
        self.assertEqual(self.manifest["canonical_parameters_sha256"], sha256(CANONICAL))
        self.assertEqual(sha256(SOURCE), sha256(CANONICAL))

    def test_contract_metadata(self):
        self.assertEqual(self.parameters["changed_layers"], 3)
        self.assertEqual(self.manifest["changed_layers"], [13, 16, 17])
        self.assertEqual(self.manifest["runtime_row_scale_policy"], "absmax")
        self.assertEqual(self.manifest["selector"], "compare_k_selectors_e2e.select_absmax")
        self.assertEqual(self.manifest["signed_int10_range"], [-512, 511])
        self.assertEqual(self.manifest["effective_shift_range"], [0, 31])
        self.assertEqual(self.manifest["status"], "FROZEN_REFERENCE")

    def test_weight_reference_is_frozen(self):
        weights = self.manifest["weight_quantization"]
        self.assertTrue(weights["unchanged_from_old_baseline"])
        self.assertEqual(weights["reference"], "diagnostics/final_downproj_ptq/")


if __name__ == "__main__":
    unittest.main()
