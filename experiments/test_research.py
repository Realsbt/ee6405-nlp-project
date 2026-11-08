import copy
import json
import itertools
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.research import validate, configure_model, optimizer_groups, auxiliary_loss
from experiments.hibernate_watchdog import decision
from experiments.hyperstack import Hyperstack


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model=nn.Module(); self.bert_model.encoder=nn.Module()
        self.bert_model.encoder.layer=nn.ModuleList([nn.Linear(4,4) for _ in range(12)])
        self.bert_model.embeddings=nn.Embedding(8,4)
        self.classifier=nn.Linear(4,4)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((Path(__file__).parent/'configs/a4000_research.json').read_text())
        self.c.pop('attention_implementation'); self.c['architecture']['bert_dim']=4
    def test_depth_and_lr_groups(self):
        for depth in (1,2,4,6,12):
            c=copy.deepcopy(self.c); c['architecture']['unfreeze_bert_layers']=depth
            model=configure_model(TinyModel(),c)
            self.assertFalse(model.bert_model.embeddings.weight.requires_grad)
            self.assertEqual([i for i,l in enumerate(model.bert_model.encoder.layer) if l.weight.requires_grad],list(range(12-depth,12)))
            groups=optimizer_groups(model,c)
            named={g['name']:g['lr'] for g in groups}
            self.assertAlmostEqual(named['encoder_11'],1.5e-5)
            self.assertAlmostEqual(named[f'encoder_{12-depth}'],1.5e-5*.85**(depth-1))
            ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
    def test_invalid_supcon_batch_and_embeddings(self):
        for batch in (4,6):
            with self.assertRaises(ValueError): validate(dict(self.c,batch_size=batch))
        with self.assertRaises(ValueError): validate(dict(self.c,unfreeze_embeddings=True))
    def test_auxiliary_targets_and_projection_gradients(self):
        c=dict(self.c,supcon_projection=True,aux_content_weight=.1,aux_edit_weight=.2)
        model=configure_model(TinyModel(),c)
        features=torch.randn(4,4,requires_grad=True); y=torch.arange(4)
        loss=auxiliary_loss(model,features,y,c)
        expected=.1*nn.functional.cross_entropy(model.content_head(features),torch.tensor([0,0,1,1]))+.2*nn.functional.cross_entropy(model.edit_head(features),torch.tensor([0,1,0,1]))
        torch.testing.assert_close(loss,expected)
        (loss+model.supcon_projection(features).square().sum()).backward()
        self.assertIsNotNone(model.supcon_projection[0].weight.grad)
        self.assertIsNotNone(features.grad)
        self.assertEqual(model.classifier(features).shape,(4,4))
    @patch('experiments.hibernate_watchdog.alive',return_value=False)
    def test_parent_dies_after_train_exit_still_bounded(self,_):
        state={'phase':'training','training_pid':123,'training_deadline':9000}
        due,idle,deadline=decision(state,100,None,600,10000)
        self.assertFalse(due); self.assertEqual(deadline,700)
        self.assertTrue(decision(state,700,idle,600,10000)[0])
    @patch('experiments.hibernate_watchdog.alive',return_value=True)
    def test_ready_and_absolute_deadline(self,_):
        s={'phase':'training','training_pid':123,'training_deadline':99999}
        self.assertTrue(decision(s,200,100,600,200)[0])
        self.assertTrue(decision(dict(s,ready_to_hibernate=True),100,None,600,99999)[0])
    def test_watchdog_retries_then_records_acceptance(self):
        from experiments.hibernate_watchdog import main
        for state in ({'phase':'complete','ready_to_hibernate':True}, {'phase':'training','training_pid':None}):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); (root/'pipeline.json').write_text(json.dumps(state))
                with patch('experiments.hibernate_watchdog.Hyperstack') as cls, \
                     patch('sys.argv',['watchdog','--batch',d,'--grace-seconds','30']), \
                     patch('experiments.hibernate_watchdog.time.time',side_effect=itertools.count(1000,1000)), \
                     patch('experiments.hibernate_watchdog.time.sleep'), \
                     patch('experiments.hibernate_watchdog.os.sync'):
                    api=cls.return_value; api.vm_id=123
                    api.preflight.return_value={'vm_id':123,'identity_verified':True}
                    api.hibernate.side_effect=[RuntimeError('network unavailable'), {'status':True}]
                    main()
                    result=json.loads((root/'hibernate-request.json').read_text())
                    self.assertEqual(result['status'],'accepted')
                    self.assertEqual(api.hibernate.call_count,2)
                    self.assertEqual(result['artifacts_complete'],state.get('ready_to_hibernate',False))
    def test_api_identity_mismatch_refuses_wrong_vm(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'key'; f.write_text('test-secret'); f.chmod(0o600)
            with patch.dict(os.environ,{'HYPERSTACK_VM_ID':'123','HYPERSTACK_API_KEY_FILE':str(f)}):
                api=Hyperstack()
                with patch.object(api,'request',return_value={'instance':{'id':123,'name':'not-this-vm','status':'ACTIVE'}}):
                    with self.assertRaisesRegex(RuntimeError,'identity'): api.preflight()
    def test_api_uses_explicit_application_user_agent(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'key'; f.write_text('test-secret'); f.chmod(0o600)
            with patch.dict(os.environ,{'HYPERSTACK_VM_ID':'123','HYPERSTACK_API_KEY_FILE':str(f)}):
                api=Hyperstack()
                with patch('experiments.hyperstack.urllib.request.urlopen') as opened:
                    opened.return_value.__enter__.return_value.read.return_value=b'{"status": true}'
                    api.request()
                    request=opened.call_args.args[0]
                    self.assertEqual(request.get_header('User-agent'),'ee6405-training/1.0')
    def test_no_push_without_gain_or_to_another_branch(self):
        from experiments.publish_results import publish_if_improved
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'score_change.json').write_text(json.dumps({'publish_eligible':False,'paired_gain_pp':-.1}))
            with patch('experiments.publish_results.subprocess.run') as run:
                self.assertEqual(publish_if_improved(root,'yuzhe')['status'],'skipped_no_confirmed_gain')
                run.assert_not_called()
            (root/'score_change.json').write_text(json.dumps({'publish_eligible':True,'paired_gain_pp':.1}))
            with patch('experiments.publish_results.subprocess.check_output',return_value='other-branch'), patch('experiments.publish_results.subprocess.run') as run:
                self.assertEqual(publish_if_improved(root,'yuzhe')['status'],'publication_failed')
                run.assert_not_called()
    def test_api_secret_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'key'; f.write_text('test-secret'); f.chmod(0o644)
            with patch.dict(os.environ,{'HYPERSTACK_VM_ID':'123','HYPERSTACK_API_KEY_FILE':str(f)}):
                with self.assertRaises(ValueError): Hyperstack()
                f.chmod(0o600); api=Hyperstack()
                with patch.object(api,'request',return_value={'status':True}) as request:
                    api.hibernate(); request.assert_called_once_with('/hibernate?retain_ip=true')


if __name__=='__main__': unittest.main()
