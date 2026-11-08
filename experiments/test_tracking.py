import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
from experiments.tracking import Tracker


class TrackingTests(unittest.TestCase):
    def make_tracker(self, root):
        run=MagicMock();run.id='test-id';run.url='https://example.invalid/run';run.project='test';run.entity='test';run.settings.mode='online';run.summary={}
        sdk=MagicMock();sdk.init.return_value=run
        with patch.dict('sys.modules',{'wandb':sdk}):tracker=Tracker(root,{'wandb_project':'test'})
        return tracker,run,sdk

    def test_durable_identity_and_validation_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            tracker,run,sdk=self.make_tracker(d)
            identity=json.loads((Path(d)/'wandb_run.json').read_text())
            self.assertEqual(identity['mode'],'online')
            metrics={'macro_f1':.8,'per_class':{'human_written':{'f1':.9,'auroc_ovr':None}},'confusion_matrix':[[1,2],[3,4]]}
            event={'epoch':2,'best_epoch':2,'stale_epochs':0,'fit_seconds':5,'validation_seconds':1,'mean_train_loss':1.,'mean_train_ce':.5,'mean_train_supcon':.5,'validation':metrics}
            tracker.event('epoch_complete',event)
            logged=run.log.call_args.args[0]
            self.assertEqual(logged['val/macro_f1'],.8)
            self.assertEqual(logged['val/human_written/f1'],.9)
            self.assertFalse(run.summary['test_partition_loaded'])
            self.assertNotIn('val/human_written/auroc_ovr',logged)
            tracker.finish('completed')
            run.finish.assert_called_once_with(exit_code=0)

    def test_network_log_error_does_not_replace_local_records(self):
        with tempfile.TemporaryDirectory() as d:
            tracker,run,_=self.make_tracker(d)
            run.log.side_effect=RuntimeError('test-network-error')
            tracker.log({'train/loss':1})
            error=json.loads((Path(d)/'tracking_errors.jsonl').read_text())
            self.assertEqual(error['type'],'RuntimeError')


if __name__=='__main__':unittest.main()
