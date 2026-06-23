#!/usr/bin/env bash
#
# Install (or update) the image-models function into a running Open WebUI.
#
#   ./install.sh <open-webui-url> <admin-api-key>
#
# Example:
#   ./install.sh http://127.0.0.1:3000 sk-xxxxxxxxxxxxxxxx
#
# Get an admin API key in Open WebUI under Settings > Account > API Keys.
# The script creates the function if missing, updates it if present, then
# activates it. After it runs, set the MODELS_JSON valve (Admin > Functions).
set -euo pipefail

URL="${1:-}"
KEY="${2:-}"
if [ -z "$URL" ] || [ -z "$KEY" ]; then
  echo "Usage: ./install.sh <open-webui-url> <admin-api-key>" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
FILE="$HERE/llamaswap_image_models.py"
FUNC_ID="llamaswap_image_models"
FUNC_NAME="Image Generation (llama-swap)"

python3 - "$URL" "$KEY" "$FILE" "$FUNC_ID" "$FUNC_NAME" <<'PY'
import sys, json, urllib.request, urllib.error

url, key, path, fid, fname = sys.argv[1:6]
url = url.rstrip('/')
content = open(path, encoding='utf-8').read()
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}

def call(method, route, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + route, data=data, headers=H, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode(errors='replace')
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')[:300]

# Does the function already exist?
status, funcs = call('GET', '/api/v1/functions/')
if status != 200:
    print("Could not list functions (%s): %s" % (status, funcs)); sys.exit(1)
exists = any(isinstance(f, dict) and f.get('id') == fid for f in (funcs or []))

form = {"id": fid, "name": fname, "content": content,
        "meta": {"description": "Image-generation models routed through llama-swap, as selectable Open WebUI models."}}

if exists:
    status, resp = call('POST', '/api/v1/functions/id/%s/update' % fid, form)
    action = "Updated"
else:
    status, resp = call('POST', '/api/v1/functions/create', form)
    action = "Created"
if status != 200:
    print("%s failed (%s): %s" % (action, status, resp)); sys.exit(1)
print("%s function '%s'." % (action, fid))

# Activate it if it is not already on.
status, fn = call('GET', '/api/v1/functions/id/%s' % fid)
if status == 200 and isinstance(fn, dict) and not fn.get('is_active'):
    s2, _ = call('POST', '/api/v1/functions/id/%s/toggle' % fid)
    print("Activated." if s2 == 200 else "Could not activate (%s)." % s2)
else:
    print("Already active.")

print("Done. Now set the MODELS_JSON valve: Open WebUI > Admin Panel > Functions > %s." % fname)
PY
