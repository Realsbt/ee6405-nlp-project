"""Serve Text Lab with document explanations, on loopback only.

Original inference.py, parser compatibility and resource paths stay unchanged.
/reading preserves the existing main Reading Studio, including excerpt tools.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

try:  # Optional: only used to report process memory in the comparison view.
    import psutil
except ImportError:  # pragma: no cover - the demo still works without it
    psutil = None


def process_memory_mib():
    """Resident memory of this server process, or None when psutil is absent."""
    if psutil is None:
        return None
    try:
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:  # pragma: no cover - never fail a prediction over telemetry
        return None


def make_handler(service, explain, page, reading_page, port, cache_stats=None, remember=None):
    """Small factory permits HTTP contract tests without loading ML checkpoints."""
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, kind="application/json; charset=utf-8"):
            payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
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
                pass  # A closed browser does not cancel a running ML forward.

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/index.html', '/explain'):
                self.respond(200, page, 'text/html; charset=utf-8')
            elif path == '/reading':
                self.respond(200, reading_page, 'text/html; charset=utf-8')
            elif path == '/api/telemetry':
                body = {'process_memory_mib': process_memory_mib()}
                if cache_stats is not None:
                    body['explanation_cache'] = cache_stats()
                self.respond(200, body)
            elif path == '/api/models':
                registry = service.registry()
                registry['explanation_api'] = 'ee6405.document-explanation/v1'
                for row in registry['models']:
                    row['supports_document_explanation'] = row['id'] in ('tfidf', 'roberta', 'race', 'race_no_supcon')
                self.respond(200, registry)
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path not in ('/api/predict', '/api/explain'):
                self.respond(404, {'error': 'Not found'})
                return
            try:
                origin = self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
                    self.respond(403, {'error': 'Use the local Demo page'})
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 50000:
                    raise ValueError('Request is empty or too large')
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Send application/json')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Send a JSON object')
                if self.path == '/api/predict':
                    result = service.predict(request.get('text'), request.get('model'))
                    if remember is not None:
                        remember(request.get('text'), request.get('model'), result)
                else:
                    result = explain(service, request.get('text'), request.get('model'),
                                     unit_ids=request.get('unit_ids'), max_units=request.get('max_units', 8),
                                     refresh=bool(request.get('refresh')), prepare_only=bool(request.get('prepare_only')))
                self.respond(200, result)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.respond(400, {'error': str(error)})
            except Exception as error:
                self.respond(500, {'error': str(error), 'type': type(error).__name__})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    from experiments.inference import DetectorService
    from experiments.document_explanations import cache_stats, explain_document, remember_document
    service = DetectorService(args.device)
    parent = Path(__file__).parent
    page = (parent / 'document_explorer.html').read_bytes()
    reading = (parent / 'demo.html').read_bytes()
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                 make_handler(service, explain_document, page, reading, args.port,
                                              cache_stats=cache_stats, remember=remember_document))
    print(f'EE6405 Text Lab: http://127.0.0.1:{args.port}/ · {args.device} · legacy tools /reading', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
