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
function staticAnswer(url,body){
 const text=body&&body.text;
 if(url==='/api/models')return STATIC_DATA.models;
 if(url==='/api/telemetry')return {process_memory_mib:null,explanation_cache:null};
 if(text!==STATIC_DATA.text)throw Error('This is a static snapshot. Only the built-in example has real model outputs; run the local service to analyze your own text.');
 const model=body.model;
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
        'Static snapshot: real model outputs for the built-in example.')
    page = page.replace(
        '<title>Text Lab — Evidence Studio</title>',
        '<title>Text Lab — Static Demo</title>')
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
