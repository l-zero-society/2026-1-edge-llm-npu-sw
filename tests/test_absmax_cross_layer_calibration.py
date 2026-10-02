import copy
import unittest

from scripts import run_absmax_cross_layer_calibration as cross


class CrossLayerTests(unittest.TestCase):
    def examples(self):
        return [dict(index=i,targets=list(range(40))) for i in range(24)]

    def test_fresh_split_and_exclusion(self):
        excluded=set(range(14));selection,final=cross.deterministic_fresh_split(self.examples(),excluded)
        self.assertEqual((len(selection),len(final)),(4,6))
        self.assertFalse(({e['index'] for e in selection}|{e['index'] for e in final})&excluded)

    def test_split_deterministic_and_disjoint(self):
        a,b=cross.deterministic_fresh_split(self.examples(),set(range(14)))
        c,d=cross.deterministic_fresh_split(copy.deepcopy(self.examples()),set(range(14)))
        self.assertEqual([x['index'] for x in a+b],[x['index'] for x in c+d])
        self.assertFalse({x['index'] for x in a}&{x['index'] for x in b})
        self.assertEqual(len(b)*cross.FINAL_TARGETS,192)

    def test_candidate_deduplication_and_old(self):
        a=dict(sources=['old'],sx=1.,s10=2.,multiplier=[3],shift=[4])
        b=dict(a,sources=['pilot']);c=dict(sources=['local_best'],sx=2.,s10=2.,multiplier=[3],shift=[4])
        result=cross.deduplicate_candidates([a,b,c])
        self.assertEqual(len(result),2);self.assertIn('old',result[0]['sources'])

    def test_sensitivity_classification(self):
        old=dict(kl=.1,nmse=.2,top1=.8)
        self.assertEqual(cross.classify_sensitivity(old,dict(kl=.09,nmse=.2,top1=.8)),'BENEFICIAL')
        self.assertEqual(cross.classify_sensitivity(old,dict(kl=.08,nmse=.22,top1=.8)),'UNSAFE')
        self.assertEqual(cross.classify_sensitivity(old,dict(kl=.11,nmse=.2,top1=.8)),'NON_BENEFICIAL')

    def test_feasibility_and_kl_primary(self):
        old=dict(kl=.1,nmse=.2,top1=.8,cosine=.9)
        self.assertTrue(cross.candidate_can_replace(old,dict(kl=.09,nmse=.21,top1=.79,cosine=.9)))
        self.assertFalse(cross.candidate_can_replace(old,dict(kl=.08,nmse=.211,top1=.8,cosine=.9)))
        self.assertFalse(cross.candidate_can_replace(old,dict(kl=.1,nmse=.19,top1=.9,cosine=.99)))

    def test_coordinate_update_and_incumbent_retention(self):
        spec={'x':dict(sx=1.,s10=1.)};frozen=copy.deepcopy(spec)
        old=dict(kl=.1,nmse=.2,top1=.8,cosine=.9)
        self.assertIsNone(cross.choose_coordinate(old,spec,[dict(candidate_id='a',specs=spec,
            metrics=dict(kl=.11,nmse=.1,top1=.9,cosine=.99))],frozen))

    def test_checkpoint_mismatch(self):
        self.assertNotEqual(cross.digest(dict(split='4/6')),cross.digest(dict(split='4/12')))

    def test_limits_and_heldout_separation(self):
        self.assertEqual(cross.MAX_SHORTLIST,6);self.assertEqual(cross.COORDINATE_PASSES,2)
        self.assertEqual((cross.SELECTION_CHATS,cross.FINAL_CHATS),(4,6))
        self.assertEqual((cross.SELECTION_TARGETS*4,cross.FINAL_TARGETS*6),(96,192))


if __name__=='__main__':unittest.main()
