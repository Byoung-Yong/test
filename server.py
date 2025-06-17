from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from simulation import simulate_cv


class CVHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="text/html"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._set_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/simulate':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                params = json.loads(body.decode())
            except Exception:
                self.send_error(400)
                return
            potentials, currents = simulate_cv(**params)
            self._set_headers(200, 'application/json')
            self.wfile.write(json.dumps({'potential': potentials,
                                         'current': currents}).encode())
        else:
            self.send_error(404)


if __name__ == '__main__':
    HTTPServer(('0.0.0.0', 8000), CVHandler).serve_forever()
