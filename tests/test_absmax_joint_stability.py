import copy
import unittest

from scripts import run_absmax_joint_stability as stability


class AbsmaxJointStabilityTests(unittest.TestCase):
    def test_fixed_subsets(self):
        self.assertEqual(stability.CAL_POSITIONS,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])
        self.assertEqual(stability.VAL_POSITIONS,[0,3,6,9,12,15,18,21,2,8,14,20])
        self.assertEqual(len(stability.CAL_POSITIONS),16)
        self.assertEqual(len(stability.VAL_POSITIONS)*stability.VAL_TARGETS,384)

    def test_fixed_search_ranges(self):
        self.assertEqual(stability.SX_JS,[-2,-1,0,1,2])
        self.assertEqual(stability.S10_IS,[-4,-3,-2,-1,0,1,2,3,4])
        self.assertEqual(len(stability.SX_JS)*len(stability.S10_IS),45)

    def test_exact_and_direction_matching(self):
        pilot=dict(layer="1",sx_j="-2",s10_i="4",sx_boundary_hit="True",
                   s10_boundary_hit="True",relative_local_nmse_change="-.1")
        current=dict(layer=1,sx_j=-1,s10_i=4,sx_boundary_hit=False,
                     s10_boundary_hit=True,relative_local_nmse_change=-.2)
        row=stability.stability_row(pilot,current)
        self.assertFalse(row["exact_sx_match"]);self.assertTrue(row["exact_s10_match"])
        self.assertFalse(row["exact_pair_match"]);self.assertTrue(row["sx_direction_match"])
        self.assertTrue(row["s10_direction_match"])

    def test_boundary_persistence(self):
        pilot=dict(layer="0",sx_j="2",s10_i="0",sx_boundary_hit="True",
                   s10_boundary_hit="False",relative_local_nmse_change="0")
        current=dict(layer=0,sx_j=2,s10_i=0,sx_boundary_hit=True,
                     s10_boundary_hit=False,relative_local_nmse_change=0)
        row=stability.stability_row(pilot,current)
        self.assertTrue(row["sx_boundary_persisted"]);self.assertFalse(row["s10_boundary_persisted"])

    def test_summary_counts(self):
        rows=[dict(exact_sx_match=True,exact_s10_match=False,exact_pair_match=False,
                   sx_direction_match=True,s10_direction_match=True,sx_boundary_persisted=False,
                   s10_boundary_persisted=True),dict(exact_sx_match=False,exact_s10_match=True,
                   exact_pair_match=False,sx_direction_match=False,s10_direction_match=True,
                   sx_boundary_persisted=True,s10_boundary_persisted=False)]
        summary=stability.stability_summary(rows)
        self.assertEqual(summary["exact_sx_match_count"],1)
        self.assertEqual(summary["s10_direction_match_count"],2)

    def test_e2e_gate_plumbing(self):
        old=dict(kl=.1,nmse=.2,top1=.8)
        self.assertTrue(stability.joint.e2e_gate(old,dict(kl=.105,nmse=.21,top1=.79))["pass"])
        self.assertFalse(stability.joint.e2e_gate(old,dict(kl=.106,nmse=.2,top1=.8))["pass"])

    def test_configuration_fingerprint(self):
        config=stability.configuration({"frozen":"x"},"pilot")
        self.assertEqual(stability.config_fingerprint(config),
                         stability.config_fingerprint(copy.deepcopy(config)))
        changed=copy.deepcopy(config);changed["seed"]+=1
        self.assertNotEqual(stability.config_fingerprint(config),stability.config_fingerprint(changed))

    def test_selector_and_policy_identity(self):
        self.assertIs(stability.joint.select_absmax,stability.pilot.select_absmax)
        self.assertEqual(stability.RESERVOIR_CAPACITY,64);self.assertEqual(stability.SEED,20261002)


if __name__=="__main__":unittest.main()
