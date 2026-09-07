import copy
import unittest

from view4.icfg_split import partition, original_split


def rows(pids, split):
    return [dict(id=pid, split=split, file_path='%d_%d.jpg'%(pid,i),
        captions=['person'], captions_bt=['a person']) for pid in pids for i in range(3)]


class SplitTests(unittest.TestCase):
    def test_identity_holdout_is_deterministic_and_does_not_edit_sources(self):
        train, test = rows(range(5),'train'), rows([7,8],'test')
        before = copy.deepcopy((train,test))
        a, manifest = partition(train,test,2)
        self.assertEqual((a,manifest),partition(train,test,2))
        self.assertEqual((train,test),before)
        ids = {s:{r['id'] for r in v} for s,v in a.items()}
        self.assertFalse(ids['train'] & ids['val'])
        self.assertFalse((ids['train']|ids['val']) & ids['test'])
        self.assertEqual(len(ids['val']),2)
        self.assertTrue(all(r['source_split']=='train' and r['split']=='val' for r in a['val']))
        self.assertTrue(all(r['source_split']=='test' for r in a['test']))

    def test_both_conflicting_records_are_quarantined(self):
        train = rows(range(4),'train')
        train[0]['file_path'] = train[3]['file_path'] = 'ambiguous.jpg'
        out, report = partition(train, rows([9],'test'),1)
        self.assertEqual(report['quarantined_paths'],{'ambiguous.jpg':[0,1]})
        self.assertEqual(report['quarantined_rows'],2)
        self.assertFalse(any(r['file_path']=='ambiguous.jpg' for v in out.values() for r in v))

    def test_overlap_and_duplicate_paths_are_rejected(self):
        train, test = rows(range(4),'train'), rows([9],'test')
        cases = [(train,rows([0],'test')),(train+[copy.deepcopy(train[0])],test)]
        collision = copy.deepcopy(test)
        collision[0]['file_path'] = train[0]['file_path']
        cases.append((train,collision))
        for a,b in cases:
            with self.assertRaises(ValueError): partition(a,b,1)

    def test_provenance_cannot_claim_holdout_is_official_val(self):
        self.assertEqual(original_split('val',{'source_split':'train'}),'train')
        for split,source in [('val','val'),('train','test'),('test','train')]:
            with self.assertRaises(ValueError): original_split(split,{'source_split':source})
