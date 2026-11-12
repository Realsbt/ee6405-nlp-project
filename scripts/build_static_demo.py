"""Build a static, self-contained Text Lab page for GitHub Pages.

The page keeps the real UI and the real model outputs for the built-in
example, but has no backend: every /api call is answered from a frozen
payload captured from the local service. Arbitrary user text cannot be
classified, and the page says so instead of pretending.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ('tfidf', 'roberta', 'race', 'race_no_supcon')
# Units measured per model so the Influence view has real numbers to show.
UNITS = {'tfidf': 6, 'roberta': 6, 'race': 8, 'race_no_supcon': 8}


def example_text() -> str:
    page = (ROOT / 'experiments/document_explorer.html').read_text(encoding='utf-8')
    match = re.search(r"const EXAMPLE='(.*?)';", page, re.S)
    if not match:
        raise SystemExit('Could not find the built-in example text')
    return match.group(1).replace('\\n', '\n')


def capture(service, text: str) -> dict:
    from experiments.document_explanations import explain_document

    payload = {'text': text, 'models': {}, 'predict': {}, 'explain': {}}
    registry = service.registry()
    registry['explanation_api'] = 'ee6405.document-explanation/v1'
    for row in registry['models']:
        row['supports_document_explanation'] = row['id'] in MODELS
    payload['models'] = registry

    for name in MODELS:
        print(f'  {name}: prediction', flush=True)
        payload['predict'][name] = service.predict(text, name)
        print(f'  {name}: metadata', flush=True)
        payload['explain'][name] = {'prepare': explain_document(service, text, name, prepare_only=True)}
        count = UNITS[name]
        print(f'  {name}: {count} units', flush=True)
        payload['explain'][name]['units'] = explain_document(
            service, text, name, max_units=count)
    return payload


def build_html(payload: dict) -> str:
    page = (ROOT / 'experiments/document_explorer.html').read_text(encoding='utf-8')
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

    shim = """
// ---- Static snapshot: every /api call is answered from a frozen payload ----
const STATIC_DATA=__STATIC_PAYLOAD__;
const STATIC_NOTE='Static snapshot · real model outputs for the built-in example only';
// ---- TF-IDF runs in the browser: fixed vocabulary, IDF and linear head ----
const TFIDF=__TFIDF_PAYLOAD__;
let tfidfReady=null;
function tfidfLoad(){
 if(tfidfReady)return tfidfReady;
 tfidfReady=(async()=>{
  const bin=s=>{const raw=atob(s);const out=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)out[i]=raw.charCodeAt(i);return out.buffer};
  const idf=new Float32Array(bin(TFIDF.idf.data));
  const coef=new Float32Array(bin(TFIDF.coef.data));
  const terms=TFIDF.terms.split('\\n');
  const index=new Map();
  for(let i=0;i<terms.length;i++)index.set(terms[i],i);
  return {idf,coef,terms,index,shape:TFIDF.coef.shape};
 })();
 return tfidfReady;
}
function tfidfTokenize(text){
 const source=TFIDF.lowercase?text.toLowerCase():text;
 // scikit-learn's pattern carries a Python-only (?u) flag; JS is Unicode by default.
 const pattern=TFIDF.token_pattern.replace('(?u)','');
 const re=new RegExp(pattern,'gu');
 const words=[];
 let m;
 while((m=re.exec(source))!==null)words.push(m[0]);
 return words;
}
async function tfidfPredict(text){
 const {idf,coef,index,shape}=await tfidfLoad();
 const words=tfidfTokenize(text);
 const [lo,hi]=TFIDF.ngram_range;
 const counts=new Map();
 for(let n=lo;n<=hi;n++){
  for(let i=0;i+n<=words.length;i++){
   const j=index.get(words.slice(i,i+n).join(' '));
   if(j!==undefined)counts.set(j,(counts.get(j)||0)+1);
  }
 }
 const vector=new Float64Array(idf.length);
 for(const [j,count] of counts)vector[j]=TFIDF.sublinear_tf?1+Math.log(count):count;
 let norm=0;
 for(let j=0;j<vector.length;j++){vector[j]*=idf[j];norm+=vector[j]*vector[j]}
 norm=Math.sqrt(norm);
 if(norm>0)for(let j=0;j<vector.length;j++)vector[j]/=norm;
 const logits=[0,0,0,0];
 for(let c=0;c<shape[0];c++){
  let sum=TFIDF.intercept[c];
  const base=c*shape[1];
  for(let j=0;j<vector.length;j++){const v=vector[j];if(v!==0)sum+=coef[base+j]*v}
  logits[c]=sum;
 }
 const max=Math.max(...logits);
 const exp=logits.map(x=>Math.exp(x-max));
 const total=exp.reduce((a,b)=>a+b,0);
 const probs=exp.map(x=>x/total);
 const covered=text.length;
 return {
  model:'tfidf',run_id:'tfidf_lr_v1',
  prediction:['human_written','human_ai_polished','ai_generated','ai_humanized'][probs.indexOf(Math.max(...probs))],
  scores:{human_written:probs[0],human_ai_polished:probs[1],ai_generated:probs[2],ai_humanized:probs[3]},
  ai_creator_score:probs[2]+probs[3],
  input:{characters:text.length,lexical_tokens_with_specials:words.length+2,
         lexical_covered_characters:covered,lexical_truncated:false,lexical_window:'full_text',
         rst_scope:null,edu_count:null},
  timing:{end_to_end_seconds:0,detector_model_load_seconds:0,detector_forward_seconds:0,
          rst_parser_initialization:null,rst_parse_seconds:null},
  device:'browser',
  scope:'Computed in your browser from the exported TF-IDF weights. Scores are not calibrated authorship probabilities.'
 };
}
function tfidfExplain(text,body){
 // Display segments mirror the server's sentence grouping.
 const segments=[];
 const re=/[.!?](?:["\u201d\u2019']+)?(?=\\s|$)|\\n\\s*\\n/g;
 const ends=[];
 let m;
 while((m=re.exec(text))!==null)ends.push(m.index+m[0].length);
 ends.push(text.length);
 let start=0;
 for(const end of [...new Set(ends)].sort((a,b)=>a-b)){
  if(end>start&&text.slice(start,end).trim()){segments.push({start,end});start=end}
 }
 if(start<text.length&&segments.length){segments[segments.length-1].end=text.length}
 const units=segments.map((s,i)=>({id:'u'+(i+1),start:s.start,end:s.end,start_utf16:s.start,end_utf16:s.end,
  status:'not_computed',without_scores:null,delta_pp:null,coverage:'full',lexical_token_count:0,
  visible_end:s.end,visible_end_utf16:s.end}));
 return {units,segments};
}
function staticAnswer(url,body){
 const text=body&&body.text;
 if(url==='/api/models')return STATIC_DATA.models;
 if(url==='/api/telemetry')return {process_memory_mib:null,explanation_cache:null};
 const model=body&&body.model;
 if(url==='/api/predict'&&model==='tfidf'&&text!==STATIC_DATA.text)return tfidfPredict(text);
 if(url==='/api/explain'&&model==='tfidf'&&text!==STATIC_DATA.text){
  return tfidfPredict(text).then(doc=>{
   const {units}=tfidfExplain(text,body);
   return {schema:'ee6405.document-explanation/v1',text_sha256:'browser',document:doc,
    method:'tfidf_occurrence_share_suppression',units_kind:'display_segments',units,graph:null,
    stages:[{stage:'tfidf_vector',tensor_shape:[1,TFIDF.n_features],active_features:0},
            {stage:'linear_logits',tensor_shape:[1,4]}],
    checked_units:0,total_units:units.length,selected_unit_ids:[],
    status:body.prepare_only?'prepared':'complete_for_requested_units',
    baseline_replay_max_abs_diff:0,restored_max_abs_diff:0,elapsed_seconds:0,
    timing:{prepare_seconds:0,units_seconds:0},cache:{preparation:'browser'},
    definition:'delta_pp = 100 × (original document score − intervened document score)',
    reference:'zero content at the explicitly named internal boundary',
    limitations:['Unit influence is not computed in the browser snapshot; only the document prediction is.']};
  });
 }
 if(text!==STATIC_DATA.text)throw Error('This is a static snapshot. Only the built-in example has real model outputs, and only TF-IDF can score new text in the browser; run the local service for RoBERTa and RACE.');
 if(url==='/api/predict')return STATIC_DATA.predict[model];
 if(url==='/api/explain'){
  const entry=STATIC_DATA.explain[model];
  if(body.prepare_only)return entry.prepare;
  const wanted=new Set(body.unit_ids||[]);
  const source=wanted.size?entry.units:entry.prepare;
  const units=source.units.map(u=>{
   if(!wanted.size)return u;
   const hit=entry.units.units.find(x=>x.id===u.id);
   return hit||u;
  });
  return {...source,units,status:wanted.size?source.status:'prepared'};
 }
 throw Error('Unknown endpoint in the static snapshot.');
}
"""
    shim = shim.replace('__STATIC_PAYLOAD__', data)
    tfidf_path = ROOT / 'docs/demo/tfidf-model.json'
    if not tfidf_path.is_file():
        raise SystemExit('Run scripts/export_tfidf_browser.py first')
    shim = shim.replace('__TFIDF_PAYLOAD__', tfidf_path.read_text(encoding='utf-8'))

    # Replace the network call with the frozen lookup.
    old_api = "async function api(url,body){const controller=new AbortController();"
    if old_api not in page:
        raise SystemExit('Could not find the api() function to replace')
    page = page.replace(old_api, shim + "\nasync function api(url,body){return staticAnswer(url,body)}\nasync function apiLive(url,body){const controller=new AbortController();", 1)

    # The static page must not claim to be a live service.
    page = page.replace(
        "$('connection').textContent='Local · '+String(data.device||'CPU').toUpperCase();",
        "$('connection').textContent='Static snapshot';")
    page = page.replace(
        '<span id="connection">Local · CPU</span>',
        '<span id="connection">Static snapshot</span>')
    page = page.replace(
        'Text stays in this local service.',
        'TF-IDF runs in your browser. RoBERTa and RACE replay the built-in example.')
    page = page.replace(
        '<title>Text Lab — Evidence Studio</title>',
        '<title>Text Lab — Static Demo</title>')
    # Say which models can score new text, so the limit is visible before a click.
    page = page.replace(
        'hint.textContent=m.requires_rst?\'Text plus discourse structure. The Structure tab shows the parsed graph.\':\'Text only. The Structure tab explains why this model has no discourse graph.\';',
        "hint.textContent=m.id==='tfidf'"
        "?'Text only. Runs in your browser, so it can score any text you paste.':"
        "(m.requires_rst?'Text plus discourse structure. The Structure tab shows the parsed graph. This snapshot replays the built-in example only.':'Text only. The Structure tab explains why this model has no discourse graph. This snapshot replays the built-in example only.');")
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'docs/demo/index.html'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--payload', default=str(ROOT / '.local/static_payload.json'))
    args = parser.parse_args()

    payload_path = pathlib.Path(args.payload)
    if payload_path.is_file():
        print(f'Reusing captured payload {payload_path}', flush=True)
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
    else:
        from experiments.inference import DetectorService
        service = DetectorService(args.device)
        try:
            print('Capturing real model outputs…', flush=True)
            payload = capture(service, example_text())
        finally:
            service.close()
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')

    html = build_html(payload)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f'Wrote {out} ({len(html)/1024:.0f} KiB)')


if __name__ == '__main__':
    main()
