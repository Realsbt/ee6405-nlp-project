"""Synthetic contract tests; these do not claim trained checkpoint accuracy."""
import copy
import json
import re
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from experiments.document_explanations import (
    LABELS, ExplanationError, Trace, check_parity, display_segments, explain_document,
    race_suppressed_features, score_array, temporary_hook, tfidf_shares,
    token_indices, utf16_offset, validate_relation_initialization, validate_units,
)


class Tokenizer:
    def __call__(self, text, **kw):
        if isinstance(text, list):
            return {'input_ids': [self(t, **kw)['input_ids'] for t in text]}
        found = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        found = found[:kw.get('max_length', 512)-2]
        ids = [0] + [(sum(map(ord, text[a:b])) % 50)+3 for a, b in found] + [2]
        return {'input_ids': ids, 'offset_mapping': [(0,0)]+found+[(0,0)], 'attention_mask':[1]*len(ids)}


class SyntheticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Module()
        self.embeddings.word_embeddings = torch.nn.Embedding(64, 6)
        self.projection = torch.nn.Linear(6,6)
    def forward(self, input_ids, attention_mask):
        x = self.embeddings.word_embeddings(input_ids)
        x = x + (x * attention_mask[...,None]).sum(1,keepdim=True) / attention_mask.sum(1)[:,None,None]
        return SimpleNamespace(last_hidden_state=torch.tanh(self.projection(x)))


class SyntheticTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.bert_model=SyntheticEncoder(); self.classifier=torch.nn.Linear(6,4)
    def forward(self, batch):
        x=self.bert_model(batch['full_text_input_ids'],batch['full_text_attention_mask']).last_hidden_state
        return {'logits':self.classifier(x[:,0]),'features':x[:,0]}


class GraphBuilder:
    def build(self, item, max_edu_nodes=None):
        edus=[n for n in item['nodes'] if n['type']=='edu_node'][:max_edu_nodes]
        rels=[n for n in item['nodes'] if n['type']=='relation_node']
        ordered=edus+rels;lookup={n['id']:i for i,n in enumerate(ordered)}
        edges=[];types_=[]
        for a,b in item['parent_child_edges']:
            if a in lookup and b in lookup:
                edges.extend([[lookup[a],lookup[b]],[lookup[b],lookup[a]]]);types_.extend([0,1])
        return SimpleNamespace(num_nodes=len(ordered),_edu_indices=list(range(len(edus))),
               _relation_indices=list(range(len(edus),len(ordered))),_edu_mapping={str(n['id']):i for i,n in enumerate(edus)},
               _root_idx=lookup.get(item['root_node_ids'][0],-1),
               edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous() if edges else torch.empty((2,0),dtype=torch.long),
               edge_type=torch.tensor(types_,dtype=torch.long))


class SyntheticConv(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(6,6)
    def forward(self, x, edges, types):
        aggregate=torch.zeros_like(x)
        if edges.numel():aggregate.index_add_(0,edges[1],x[edges[0]]*.2)
        return self.linear(x+aggregate)


class SyntheticRace(torch.nn.Module):
    """Matches the pinned *feature initialization* contract, not RGCN weights."""
    def __init__(self):
        super().__init__();self.bert_model=SyntheticEncoder();self.node_type_embedding=torch.nn.Embedding(2,6)
        self.node_proj=torch.nn.Linear(6,6);self.convs=torch.nn.ModuleList([SyntheticConv(),SyntheticConv()]);self.classifier=torch.nn.Linear(6,4)
        self.graph_builder=GraphBuilder();self.config={'pooling_strategy':'root'}
        self.edge_types_list=[('relation_node','Elaboration','edu_node'),('edu_node','rev_Elaboration','relation_node')]
    def forward(self,b):
        hidden=self.bert_model(b['full_text_input_ids'],b['full_text_attention_mask']).last_hidden_state[0]
        item={k:b[k][0] for k in ('nodes','parent_child_edges','root_node_ids','descendant_map','edu_token_spans')}
        graph=self.graph_builder.build(item,len(item['edu_token_spans']))
        edu=torch.stack([hidden[a:z].mean(0) for a,z in item['edu_token_spans']])
        rels=[n for n in item['nodes'] if n['type']=='relation_node'];relation=[]
        for n in rels:
            idx=[graph._edu_mapping[str(i)] for i in item['descendant_map'][n['id']] if str(i) in graph._edu_mapping]
            relation.append(edu[idx].mean(0) if idx else torch.zeros(6))
        unified=torch.cat([edu+self.node_type_embedding.weight[0],torch.stack(relation)+self.node_type_embedding.weight[1]])
        x=self.node_proj(unified)
        for i,c in enumerate(self.convs):
            x=c(x,graph.edge_index,graph.edge_type)
            if i==0:x=torch.nn.functional.elu(x)
        features=x[graph._root_idx].unsqueeze(0)
        return {'features':features,'logits':self.classifier(features)}


def text_item(text,tok):
    ids=tok(text,max_length=512)['input_ids']
    return {'full_text_input_ids':torch.tensor(ids),'full_text_attention_mask':torch.ones(len(ids),dtype=torch.long),
            'label':torch.tensor(-1),'id':'fixture'}


def race_item(text,tok):
    result=text_item(text,tok);cut=text.index('.')+1
    spans=[(0,cut),(cut,len(text))]
    offsets=tok(text,max_length=512)['offset_mapping']
    nodes=[{'id':0,'type':'relation_node','relation':'Elaboration','nuclearity':'NN'}]
    tokens=[]
    for i,(a,b) in enumerate(spans,1):
        nodes.append({'id':i,'type':'edu_node','start':a,'end':b,'relation':'elementary'})
        idx=token_indices(offsets,a,b)
        if idx:tokens.append([idx[0],idx[-1]+1])
    result.update(nodes=nodes,edu_token_spans=tokens,descendant_map={0:[1,2]},parent_child_edges=[(0,1),(0,2)],root_node_ids=[0])
    return result


def collate(items):
    assert len(items)==1
    item=items[0]
    b={k:item[k].unsqueeze(0) for k in ('full_text_input_ids','full_text_attention_mask')}
    for k in ('nodes','edu_token_spans','descendant_map','parent_child_edges','root_node_ids'):
        if k in item:b[k]=[item[k]]
    return b


class Service:
    def __init__(self):
        torch.manual_seed(19);self.models={'roberta':SyntheticTextModel().eval(),'race':SyntheticRace().eval()}
        self.models['race_no_supcon']=copy.deepcopy(self.models['race'])
        texts=['Human people compose essays. Personal stories make sense.','Software generates automated output. Machines write content.',
               'Writers polish their essay wording. The revision sounds formal.','Humanized machine content is rewritten. Stories seem personal.']*3
        y=[0,2,1,3]*3
        self.models['tfidf']=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('classifier',LogisticRegression(max_iter=100))]).fit(texts,y)
        self.tokenizer=Tokenizer();self.lock=threading.Lock();self.device='cpu';self.predict_calls=0
    def _load_model(self,name):return self.models[name],0.0
    def predict(self,text,name):
        self.predict_calls+=1
        with self.lock:
            enc=self.tokenizer(text,max_length=512);end=max(z for a,z in enc['offset_mapping'])
            if name=='tfidf':p=self.models[name].predict_proba([text[:end]])[0]
            else:
                item=race_item(text,self.tokenizer) if name.startswith('race') else text_item(text,self.tokenizer)
                with torch.inference_mode():p=self.models[name](collate([item]))['logits'].float().softmax(-1)[0].numpy()
            d={'model':name,'run_id':'SYNTHETIC_CONTRACT_ONLY','prediction':LABELS[int(np.argmax(p))],
               'scores':dict(zip(LABELS,map(float,p))),'input':{'characters':len(text),'lexical_covered_characters':end,
               'lexical_tokens_with_specials':len(enc['input_ids']),'lexical_truncated':end<len(text.rstrip())},
               'timing':{'end_to_end_seconds':0.01},'device':'cpu'}
            if name.startswith('race'):d['rst_tree']={'relation':'SYNTHETIC_TREE_FIXTURE'}
            return d


@contextmanager
def fake_data_interfaces():
    neural=types.ModuleType('experiments.neural');raw=types.ModuleType('experiments.raw_graph')
    class Text:
        def __init__(self,records,tokenizer):self.item=text_item(records[0]['article'],tokenizer)
        def __getitem__(self,i):return self.item
    class Graph:
        def __init__(self,records,tokenizer):self.item=race_item(records[0]['article'],tokenizer)
        def __getitem__(self,i):return self.item
    neural.TextDataset=Text;neural.collate_batch=collate;raw.RawGraphDataset=Graph
    with patch.dict(sys.modules,{'experiments.neural':neural,'experiments.raw_graph':raw}):yield


class CoreTests(unittest.TestCase):
    def test_segment_exact_coverage(self):
        text='First sentence.\n\nSecond sentence! Last part';u=display_segments(text);validate_units(u,text)
        self.assertEqual(''.join(text[x['start']:x['end']] for x in u),text)
    def test_emoji_offsets(self):
        self.assertEqual(utf16_offset('a🧪b',2),3)
        for x in display_segments('A 🧪 result. Next sentence.'):
            self.assertGreaterEqual(x['end_utf16'],x['end'])
    def test_special_tokens_excluded(self):self.assertEqual(token_indices([(0,0),(0,4),(5,8),(0,0)],0,4),[1])
    def test_bad_probability(self):
        for v in ([1,1,0,0],[float('nan'),0,0,0],[-.1,.5,.5,.1]):
            with self.assertRaises(ExplanationError):score_array(v)
    def test_parity_refuses_change(self):
        with self.assertRaises(ExplanationError):check_parity([.7,.1,.1,.1],[.1,.7,.1,.1])
    def test_hook_cleanup_on_exception(self):
        m=torch.nn.Linear(2,2);before=len(m._forward_hooks)
        with self.assertRaises(RuntimeError):
            with temporary_hook(m,lambda *a:None):raise RuntimeError('synthetic')
        self.assertEqual(len(m._forward_hooks),before)
    def test_nested_relation_suppression_and_types(self):
        t0=torch.tensor([10.,20.]);t1=torch.tensor([30.,40.]);edu=torch.tensor([[2.,4.],[6.,8.],[10.,12.]])
        graph=SimpleNamespace(num_nodes=5,_edu_indices=[0,1,2],_relation_indices=[3,4],_edu_mapping={'1':0,'3':1,'4':2})
        nodes=[{'id':0,'type':'relation_node'},{'id':1,'type':'edu_node'},{'id':2,'type':'relation_node'},
               {'id':3,'type':'edu_node'},{'id':4,'type':'edu_node'}]
        desc={0:[1,3,4],2:[3,4]};unified=torch.cat([edu+t0,torch.stack([edu.mean(0),edu[1:].mean(0)])+t1]);saved=unified.clone()
        validate_relation_initialization(unified,torch.stack([t0,t1]),graph,nodes,desc)
        changed,touched=race_suppressed_features(unified,t0,graph,nodes,desc,3)
        self.assertTrue(torch.equal(unified,saved));self.assertTrue(torch.equal(changed[1],t0));self.assertTrue(torch.equal(changed[0],unified[0]))
        self.assertTrue(torch.allclose(changed[3],unified[3]-edu[1]/3));self.assertTrue(torch.allclose(changed[4],unified[4]-edu[1]/2));self.assertEqual(touched,[1,3,4])
    def test_relation_contract_rejects_wrong_mean(self):
        g=SimpleNamespace(_edu_indices=[0],_relation_indices=[1],_edu_mapping={'1':0})
        with self.assertRaises(ExplanationError):validate_relation_initialization(torch.ones(2,2),torch.zeros(2,2),g,[{'id':0,'type':'relation_node'}],{0:[]})
    def test_tfidf_shares_conserve_repeated_feature(self):
        text='Same word. Same word.';v=TfidfVectorizer(ngram_range=(1,2)).fit([text]);u=display_segments(text);shares=tfidf_shares(v,text,u)
        for j in v.vocabulary_.values():self.assertAlmostEqual(sum(row.get(j,0) for row in shares.values()),1.)
    def test_tfidf_unsupported_custom_preprocessor(self):
        v=TfidfVectorizer(preprocessor=str.upper).fit(['Some valid words'])
        with self.assertRaises(ExplanationError):tfidf_shares(v,'Some valid words',display_segments('Some valid words'))
    def test_span_overlap_rejected(self):
        with self.assertRaises(ExplanationError):validate_units([{'start':0,'end':4},{'start':3,'end':5}],'abcde')


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):self.service=Service();self.text='Human writers compose a personal story. Software can polish the final wording.'
    def run_case(self,name):
        model=self.service.models[name]
        weights={k:v.clone() for k,v in model.state_dict().items()} if name!='tfidf' else None
        with fake_data_interfaces():out=explain_document(self.service,self.text,name,max_units=2)
        self.assertEqual(out['checked_units'],2);self.assertLessEqual(out['restored_max_abs_diff'],2e-5)
        self.assertEqual(self.service.predict_calls,1) # no per-passage predict API; parser not rerun
        for u in out['units']:
            self.assertEqual(u['status'],'computed')
            for c in LABELS:self.assertAlmostEqual(u['delta_pp'][c],100*(out['document']['scores'][c]-u['without_scores'][c]),places=5)
        if weights:
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))
            for m in model.modules():self.assertFalse(m.training);self.assertEqual(len(m._forward_hooks),0);self.assertEqual(len(m._forward_pre_hooks),0)
        json.dumps(out,allow_nan=False)
        return out
    def test_roberta_full_document_probe(self):self.run_case('roberta')
    def test_race_feature_probe(self):
        out=self.run_case('race');self.assertEqual(len(out['graph']['nodes']),3)
        self.assertEqual(len(out['graph']['edges']),4)
        for u in out['units']:self.assertEqual(len(u['directly_affected_node_indices']),2)
    def test_race_no_supcon_same_interface(self):self.run_case('race_no_supcon')
    def test_tfidf_feature_probe(self):self.run_case('tfidf')
    def test_subset_request(self):
        with fake_data_interfaces():out=explain_document(self.service,self.text,'race',unit_ids=['u2'],max_units=1)
        self.assertEqual(out['checked_units'],1);self.assertEqual(out['units'][0]['status'],'not_computed')
    def test_invalid_method_and_budget(self):
        for kwargs in ({'name':'notreal'},{'name':'race','max_units':9},{'name':'race','max_units':True},{'name':'race','unit_ids':['u1','u1']}):
            with self.assertRaises(ExplanationError):explain_document(self.service,self.text,**kwargs)
    def test_unknown_unit_refused(self):
        with fake_data_interfaces(),self.assertRaises(ExplanationError):explain_document(self.service,self.text,'race',unit_ids=['u999'])
    def test_truncation_never_assigns_outside_influence(self):
        text='word '*520+'. Final sentence.'
        with fake_data_interfaces():out=explain_document(self.service,text,'roberta')
        self.assertEqual(out['units'][0]['coverage'],'partial')
        self.assertEqual(out['units'][-1]['status'],'outside_lexical_window')
        self.assertIsNone(out['units'][-1]['delta_pp'])
    def test_budget_stops_between_units(self):
        with fake_data_interfaces(),patch('experiments.document_explanations.MAX_ANALYSIS_SECONDS',0):
            out=explain_document(self.service,self.text,'race',max_units=2)
        self.assertEqual(out['checked_units'],1);self.assertEqual(out['status'],'budget_stopped')
        self.assertEqual(out['units'][1]['status'],'not_computed')
    def test_intervention_error_cleans_up_hooks(self):
        model=self.service.models['race'];calls=[0]
        def fail_later(m,a,o):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('synthetic failure in intervention')
        h=model.convs[0].register_forward_hook(fail_later)
        try:
            with fake_data_interfaces(),self.assertRaises(RuntimeError):explain_document(self.service,self.text,'race')
        finally:h.remove()
        for m in model.modules():self.assertEqual(len(m._forward_hooks),0);self.assertEqual(len(m._forward_pre_hooks),0)
    def test_training_model_refused(self):
        self.service.models['roberta'].train()
        with fake_data_interfaces(),self.assertRaises(ExplanationError):explain_document(self.service,self.text,'roberta')
    def test_prediction_parity_failure_does_not_mutate(self):
        original=self.service.predict
        def wrong(*args):
            d=original(*args);d['scores']=dict(zip(LABELS,[.97,.01,.01,.01]));return d
        self.service.predict=wrong
        with fake_data_interfaces(),self.assertRaises(ExplanationError):explain_document(self.service,self.text,'race')
        for m in self.service.models['race'].modules():self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)

if __name__=='__main__':unittest.main()
