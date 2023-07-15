#!/usr/bin/env python3
import sys
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(sys.argv[1]) if len(sys.argv) > 1 else None
    HTTPServer(('', 8000), CORSRequestHandler).serve_forever()
