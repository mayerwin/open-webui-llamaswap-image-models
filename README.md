# Open WebUI image models via llama-swap

Add image-generation models to [Open WebUI](https://github.com/open-webui/open-webui) as **selectable entries in the model dropdown**. Pick one, type a prompt, get an image in the chat. Each model is routed to the right backend **through [llama-swap](https://github.com/mostlygeek/llama-swap)**, so a single GPU is shared safely between your chat LLMs and your image models (llama-swap loads one model at a time, so no out-of-memory).

![Model selector](screenshots/model-selector.png)

This is a single Open WebUI **Function** (a manifold "pipe"). No fork, no extra service, no second inference engine.

## Why

Open WebUI's built-in image generation targets one global model at a time, and you cannot choose the model per chat. If you serve several image models behind llama-swap (FLUX, Qwen-Image, HiDream, an Ideogram-style HTTP backend, and so on), this function exposes each of them as its own selectable model and routes the request to the correct backend. Your chat history becomes your gallery.

## How it works

```
you pick a model + prompt
        |
        v
   Open WebUI  --(this function)-->  llama-swap  -->  the right backend
                                                  |
                                                  +-- ComfyUI (builds a workflow)
                                                  |     POST /upstream/<name>/prompt
                                                  |
                                                  +-- OpenAI-style image API
                                                        POST /upstream/<name>/v1/images/generations
```

- **`comfyui` backend**: the function builds a ComfyUI prompt graph from a built-in template (`flux2`, `qwen_image`, `hidream`), posts it to `<llama-swap>/upstream/<name>/prompt`, polls `/history`, and fetches the image from `/view`.
- **`openai` backend**: the function posts `{prompt, size}` to `<llama-swap>/upstream/<name>/v1/images/generations` (any OpenAI-images-compatible server behind llama-swap).

The image is returned inline (a base64 data URI), so it is saved with the chat.

## Requirements

- [Open WebUI](https://github.com/open-webui/open-webui) 0.5 or newer.
- [llama-swap](https://github.com/mostlygeek/llama-swap) in front of your image backends, with each image model reachable at `/upstream/<model-name>/...`. ComfyUI is run as a llama-swap model; OpenAI-style image servers too. This keeps one GPU coordinated across chat and image models.

You do not need any extra Python packages: the function only uses `aiohttp` and `pydantic`, which Open WebUI already bundles.

## Install

### Option A: paste into Open WebUI (easiest)

1. Open Open WebUI, then **Admin Panel** > **Functions** > **+** (New Function).
2. Give it an id (for example `llamaswap_image_models`) and a name, then paste the contents of [`llamaswap_image_models.py`](llamaswap_image_models.py).
3. Save, then toggle the function **on**.
4. Configure it (see the next section). The models appear in the model dropdown.

### Option B: one command (API)

```bash
git clone https://github.com/mayerwin/open-webui-llamaswap-image-models
cd open-webui-llamaswap-image-models
./install.sh http://YOUR-OPEN-WEBUI:3000 YOUR_ADMIN_API_KEY
```

Get an API key in Open WebUI under **Settings** > **Account** > **API Keys** (the account must be an admin). The script creates the function if it is missing, updates it if it already exists, and activates it.

## Configure

In Open WebUI go to **Admin Panel** > **Functions** > your function > the **gear** icon (Valves):

- **`LLAMASWAP_URL`**: your llama-swap base URL (default `http://127.0.0.1:8080`).
- **`MODELS_JSON`**: the list of image models to expose. Replace the example with yours.
- **`TIMEOUT_S`**: max seconds to wait for one image (heavy models are slow; default 2400).

Each `MODELS_JSON` entry looks like this:

```json
{
  "id": "flux",
  "name": "FLUX.2 (fast)",
  "icon": "🎨",
  "backend": "comfyui",
  "upstream": "comfyui",
  "template": "flux2",
  "files": {
    "unet": "flux-2-klein-4b-Q4_K_M.gguf",
    "clip": "qwen_3_4b_fp4_flux2.safetensors",
    "vae":  "flux2-vae.safetensors"
  },
  "steps": 20,
  "guidance": 3.5,
  "description": "Describe the picture you want."
}
```

Field by field:

| field | meaning |
| --- | --- |
| `id` | unique id; becomes the model id in Open WebUI |
| `name` | label shown in the dropdown |
| `icon` | optional prefix (any emoji or text) |
| `backend` | `comfyui` or `openai` |
| `upstream` | the llama-swap model name to route to |
| `template` | `comfyui` only: `flux2`, `qwen_image`, or `hidream` |
| `files` | `comfyui` only: the file names ComfyUI loads (keys depend on the template) |
| `steps`, `guidance` | `comfyui` only: defaults for this model |
| `description` | shown on the model's splash screen |

An `openai` backend entry is just the routing:

```json
{ "id": "ideogram", "name": "Ideogram 4", "icon": "🎨", "backend": "openai", "upstream": "ideogram-4" }
```

The built-in templates (`flux2`, `qwen_image`, `hidream`) cover common ComfyUI graphs. For a different model family, add a builder function near the top of the source and reference it by name from `template`.

### Tidy up (optional)

Disable Open WebUI's built-in image toggle so it does not compete with these models: **Admin Panel** > **Settings** > **Images** > turn **Image Generation** off. This function does not use it.

## Use

Pick a model in the dropdown, type a description, and send. Optional inline flags in the prompt:

```
a red fox in a snowy forest --steps 30 --size 1216x832 --seed 7 --neg "blurry, low quality"
```

Or set per-user defaults in the function's UserValves (width, height, steps, seed, negative).

## Notes and troubleshooting

- **Open WebUI may run Python older than 3.12.** There, a backslash inside an f-string expression is a `SyntaxError`, and it crashes the function load with HTTP 400. If you edit the source, keep escapes (like the emoji) in module constants, never inline in an f-string. This repo already follows that rule.
- **Heavy models are slow.** A large diffusion model on a modest GPU can take many minutes per image; the function streams a "rendering... Ns" status while it waits. Raise `TIMEOUT_S` if a model needs longer.
- **The image is returned inline (base64).** Saving to Open WebUI's file store was tried, but its `/api/v1/files/` endpoint runs a document-processing pipeline that hangs on an image, so the function returns a data URI instead.
- **Nothing in the dropdown?** Check that the function is toggled on, that `MODELS_JSON` is valid JSON, and that each entry has an `id`.

## License

MIT. See [LICENSE](LICENSE).
