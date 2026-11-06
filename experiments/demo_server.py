"""Serve the local interactive course Demo on loopback only."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from experiments.inference import DetectorService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    service = DetectorService(args.device)
    page = (Path(__file__).parent / "demo.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, kind="application/json; charset=utf-8"):
            payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.respond(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/models":
                self.respond(200, service.registry())
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/predict":
                self.respond(404, {"error": "Not found"})
                return
            try:
                origin = self.headers.get("Origin")
                if origin and origin not in (f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                    self.respond(403, {"error": "Use the local Demo page"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 50000:
                    raise ValueError("Request is empty or too large")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send application/json")
                request = json.loads(self.rfile.read(length))
                result = service.predict(request.get("text"), request.get("model"))
                self.respond(200, result)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception as error:
                self.respond(500, {"error": str(error), "type": type(error).__name__})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"EE6405 Demo: http://127.0.0.1:{args.port}/ · detector {args.device} · RST parser CPU", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        server.server_close()


if __name__ == "__main__":
    main()
