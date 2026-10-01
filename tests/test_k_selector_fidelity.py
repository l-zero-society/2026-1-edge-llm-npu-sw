import copy
import unittest
import numpy as np
from scripts import confirm_k_selector_fidelity as c

class FidelityTests(unittest.TestCase):
    def test_subset(self): self.assertEqual(c.SUBSET,[0,3,6,9,12,15,18,21])
    def test_population(self): self.assertEqual(len(c.SUBSET)*c.TARGETS,256)
    def test_decision_boundaries(self):
        ms=dict(kl=.1,nmse=.2,top1=.9)
        self.assertEqual(c.decision(ms,dict(kl=.11,nmse=.22,top1=.89))['overall'],'HW-ready')
        for key,value in [('kl',.110001),('nmse',.220001),('top1',.889999)]:
            hw=dict(kl=.11,nmse=.22,top1=.89);hw[key]=value
            self.assertEqual(c.decision(ms,hw)['overall'],'not-yet')
    def test_selector_identity(self): self.assertIs(c.select_absmax,c.compare.select_absmax)
    def test_independent_caches(self):
        a,b,d=object(),object(),object();c.assert_independent([a,b,d])
        with self.assertRaises(AssertionError): c.assert_independent([a,b,a])
    def test_deltas(self):
        out=c.deltas(dict(kl=2.,nmse=4.,ppl=10.,top1=.8),dict(kl=3.,nmse=2.,ppl=9.,top1=.81))
        for key,expected in dict(kl_relative=.5,nmse_relative=-.5,ppl_relative=-.1,top1_pp=1.).items():
            self.assertAlmostEqual(out[key],expected)
    def test_fingerprint(self):
        s={'x':dict(sx=.1,s10=.2,wq=np.array([[1]],np.int8),params=dict(multiplier=np.array([3]),shift=np.array([5])))}
        original=c.frozen_fingerprint(s,b'{}')
        self.assertEqual(original,c.frozen_fingerprint(copy.deepcopy(s),b'{}'))
        self.assertNotEqual(original,c.frozen_fingerprint(s,b'{ }'))
        for key in ['sx','s10','wq','multiplier','shift']:
            ss=copy.deepcopy(s)
            if key in ['sx','s10']:ss['x'][key]*=2
            elif key=='wq':ss['x']['wq'][0,0]+=1
            else:ss['x']['params'][key][0]+=1
            self.assertNotEqual(original,c.frozen_fingerprint(ss,b'{}'))

if __name__=='__main__':unittest.main()
