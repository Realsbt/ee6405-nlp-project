"""Serve Text Lab with only the TF-IDF model, for a small VPS.

The full demo needs torch and a 500 MB transformer per model, which does not
fit comfortably in 1 GiB of RAM. TF-IDF needs joblib and numpy only: about
130 MiB resident, and roughly 1 ms per prediction.

This server exposes the same /api/predict contract as demo_server.py for the
tfidf model, and reports the other models as unavailable so the page disables
them instead of failing at request time.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np

LABELS = ('human_written', 'human_ai_polished', 'ai_generated', 'ai_humanized')
RUN_ID = 'tfidf_lr_v1'
MAX_CHARACTERS = 12000


def storage_root() -> Path:
    configured = os.environ.get('EE6405_STORAGE')
    return Path(configured) if configured else Path.home() / 'ee6405-resources'


def load_model():
    """Load the verified TF-IDF pipeline and check its recorded hash."""
    import hashlib

    from experiments.paths import resolve_recorded_path

    report = Path(__file__).resolve().parents[1] / 'experiments/reports' / RUN_ID / 'result.json'
    result = json.loads(report.read_text(encoding='utf-8'))
    # Historical results record an absolute path from the original machine;
    # resolve_recorded_path maps that prefix onto the current EE6405_STORAGE.
    path = resolve_recorded_path(result['model_path'])
    if not path.is_file():
        raise SystemExit(f'TF-IDF model not found at {path}; set EE6405_STORAGE')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != result['model_sha256']:
        raise SystemExit('TF-IDF model hash does not match the recorded result')
    return joblib.load(path)


def predict(model, text: str) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARACTERS:
        raise ValueError(f'Enter non-empty English text, up to {MAX_CHARACTERS:,} characters.')
    scores = model.predict_proba([text])[0]
    if not np.isfinite(scores).all() or abs(float(scores.sum()) - 1) > 1e-5:
        raise ValueError('The model returned an invalid score vector')
    return {
        'model': 'tfidf',
        'run_id': RUN_ID,
        'prediction': LABELS[int(np.argmax(scores))],
        'scores': dict(zip(LABELS, map(float, scores))),
        'ai_creator_score': float(scores[2] + scores[3]),
        'input': {
            'characters': len(text),
            'lexical_tokens_with_specials': None,
            'lexical_covered_characters': len(text),
            'lexical_truncated': False,
            'lexical_window': 'full_text',
            'rst_scope': None,
            'edu_count': None,
        },
        'timing': {
            'end_to_end_seconds': 0.0,
            'detector_model_load_seconds': 0.0,
            'detector_forward_seconds': 0.0,
            'rst_parser_initialization': None,
            'rst_parse_seconds': None,
        },
        'device': 'cpu',
        'scope': ('English HART-trained four-class model; class scores are not calibrated '
                  'authorship probabilities.'),
    }


def registry() -> dict:
    return {
        'models': [
            {'id': 'tfidf', 'name': 'TF-IDF + Logistic Regression', 'available': True,
             'availability_reason': '', 'requires_rst': False, 'run_id': RUN_ID},
            {'id': 'roberta', 'name': 'RoBERTa', 'available': False,
             'availability_reason': 'Not installed on this server', 'requires_rst': False,
             'run_id': 'roberta_s3407'},
            {'id': 'race', 'name': 'RACE', 'available': False,
             'availability_reason': 'Not installed on this server', 'requires_rst': True,
             'run_id': 'race_s3407'},
            {'id': 'race_no_supcon', 'name': 'RACE · no contrastive loss', 'available': False,
             'availability_reason': 'Not installed on this server', 'requires_rst': True,
             'run_id': 'race_no_supcon_s3407'},
        ],
        'device': 'cpu',
        'max_characters': MAX_CHARACTERS,
        'labels': list(LABELS),
        'parser_loaded': False,
        'explanation_api': None,
    }


def make_handler(model, page: bytes, allowed_origins: set[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'TextLabTFIDF/1.0'

        def respond(self, code, body, kind='application/json; charset=utf-8'):
            payload = body if isinstance(body, bytes) else json.dumps(
                body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/index.html'):
                self.respond(200, page, 'text/html; charset=utf-8')
            elif path == '/api/models':
                self.respond(200, registry())
            elif path == '/api/health':
                self.respond(200, {'status': 'ok', 'model': 'tfidf'})
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/predict':
                self.respond(404, {'error': 'Not found'})
                return
            try:
                origin = self.headers.get('Origin')
                if origin and allowed_origins and origin not in allowed_origins:
                    self.respond(403, {'error': 'Origin not allowed'})
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 50000:
                    raise ValueError('Request is empty or too large')
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Send application/json')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Send a JSON object')
                if request.get('model') != 'tfidf':
                    raise ValueError('This server only runs the TF-IDF model')
                self.respond(200, predict(model, request.get('text')))
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.respond(400, {'error': str(error)})
            except Exception as error:  # pragma: no cover - never leak a traceback
                self.respond(500, {'error': str(error), 'type': type(error).__name__})

        def log_message(self, fmt, *args):
            print(f'{self.address_string()} {fmt % args}', flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--page', default=str(Path(__file__).parent / 'document_explorer.html'))
    parser.add_argument('--allow-origin', action='append', default=[],
                        help='Extra allowed Origin, e.g. https://example.com')
    args = parser.parse_args()

    model = load_model()
    page = Path(args.page).read_bytes()
    origins = {f'http://{args.host}:{args.port}', f'http://localhost:{args.port}'}
    origins.update(args.allow_origin)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, page, origins))
    print(f'Text Lab (TF-IDF only): http://{args.host}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
