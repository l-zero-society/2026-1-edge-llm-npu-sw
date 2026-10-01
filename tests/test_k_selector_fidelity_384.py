import copy
import unittest
import numpy as np
from scripts import confirm_k_selector_fidelity_384 as c

class Fidelity384Tests(unittest.TestCase):
    def test_subset(self):self.assertEqual(c.SUBSET,[0,3,6,9,12,15,18,21,2,8,14,20])
    def test_old_prefix(self):self.assertEqual(c.SUBSET[:8],c.OLD_SUBSET)
    def test_targets(self):self.assertEqual(len(c.SUBSET)*c.TARGETS,384)
    def test_selector_identity(self):self.assertIs(c.select_absmax,c.prior.compare.select_absmax)
    def test_independent_caches(self):
        a,b,d=object(),object(),object();c.prior.assert_independent([a,b,d])
        with self.assertRaises(AssertionError):c.prior.assert_independent([a,b,a])
    def test_deltas(self):
        d=c.prior.deltas(dict(kl=2.,nmse=4.,ppl=10.,top1=.8),dict(kl=3.,nmse=2.,ppl=9.,top1=.81))
        for k,v in dict(kl_relative=.5,nmse_relative=-.5,ppl_relative=-.1,top1_pp=1.).items():self.assertAlmostEqual(d[k],v)
    def test_one_sided_boundary(self):
        ms=dict(kl=.1,nmse=.2,top1=.9)
        for top1 in [.89,.95]:self.assertEqual(c.decision(ms,dict(kl=.11,nmse=.22,top1=top1))['overall'],'HW-ready')
        self.assertFalse(c.decision(ms,dict(kl=.11,nmse=.22,top1=.889999))['top1'])
    def test_classification(self):
        ms=dict(kl=.1,nmse=.2,top1=.9)
        for kl,nmse,top1,case in [(.11,.22,.89,'A'),(.115,.2,.91,'B'),(.116,.2,.91,'C'),(.105,.23,.91,'C'),(.105,.2,.88,'C')]:
            self.assertEqual(c.classify(ms,dict(kl=kl,nmse=nmse,top1=top1))['case'],case)
    def test_bootstrap_determinism(self):
        rows=[dict(delta_kl=i/100,delta_nmse=-i/1000,delta_top1=i/200) for i in range(12)]
        a=c.bootstrap(rows);self.assertEqual(a,c.bootstrap(rows));self.assertEqual(a['resamples'],10000)
        for key in ['kl','nmse','top1']:self.assertLess(a['intervals'][key]['low'],a['intervals'][key]['high'])
    def test_fingerprint_stability(self):
        s={'x':dict(sx=.1,s10=.2,wq=np.array([[1]],np.int8),params=dict(multiplier=np.array([3]),shift=np.array([5])))}
        a=c.prior.frozen_fingerprint(s,b'{}');self.assertEqual(a,c.prior.frozen_fingerprint(copy.deepcopy(s),b'{}'))
        s['x']['sx']=.2;self.assertNotEqual(a,c.prior.frozen_fingerprint(s,b'{}'))

if __name__=='__main__':unittest.main()
