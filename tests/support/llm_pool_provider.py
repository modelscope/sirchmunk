"""Opt-in, loopback-only OpenAI protocol fixture. No prompts/keys are logged.

Used for browser/package acceptance; never loaded by the production app.
"""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

lock = threading.Lock()
calls, active, peaks = Counter(), Counter(), Counter()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/stats":
            self.send_json(404, {"error": "not found"})
            return
        with lock:
            self.send_json(200, {"calls": dict(calls), "active": dict(active), "peaks": dict(peaks)})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 128 * 1024:
            self.send_json(413, {"error": "request size"})
            return
        body = json.loads(self.rfile.read(size))
        model = str(body.get("model", "fixture"))[:100]
        with lock:
            calls[model] += 1
            active[model] += 1
            peaks[model] = max(peaks[model], active[model])
        try:
            if "fail" in model:
                self.send_json(503, {"error": {"message": "synthetic unavailable", "type": "server_error"}})
                return
            if "rate" in model:
                self.send_json(429, {"error": {"message": "synthetic rate limit", "type": "rate_limit_error"}})
                return
            time.sleep(1.5 if "timeout" in model else 1 if "slow" in model else .03)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if "interrupt" in model:
                    self.send_header("Content-Length", "1048576")
                self.end_headers()
                for content in ("Fixture ", model, " OK"):
                    chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0, "model": model,
                             "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                    if "interrupt" in model:
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.close_connection = True
                        return
                chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0, "model": model,
                         "choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
            else:
                self.send_json(200, {"id": "fixture", "object": "chat.completion", "created": 0, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with lock:
                active[model] -= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"fixture_port": server.server_port}), flush=True)
    server.serve_forever()
