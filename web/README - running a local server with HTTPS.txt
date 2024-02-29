You must first run a command to generate a self-signed certificate, e.g. on OS X:

openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem

Then you can run the web server with HTTPS like so:

python3 local_server_https.py


Find the IP address of this computer (the one running the server).
On a Quest or Vision Pro (it MUST be on the same LAN), go to the following URL in the browser:

<ip address>:443/index.html

