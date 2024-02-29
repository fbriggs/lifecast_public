#!/usr/bin/env python3
import sys
import os
import ssl
from http.server import SimpleHTTPRequestHandler, HTTPServer

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(sys.argv[1]) if len(sys.argv) > 1 else None
    server_address = ('', 443)  # Using standard HTTPS port

    httpd = HTTPServer(server_address, CORSRequestHandler)

    # Set up SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain('certificate.pem', 'key.pem')
    except Exception as e:
        print(f"Failed to load SSL certificate: {e}")
        sys.exit(1)

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    try:
        print("Serving HTTPS on port 443...")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")
