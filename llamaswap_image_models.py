"""
title: Image Generation (llama-swap)
author: mayerwin
author_url: https://github.com/mayerwin/open-webui-llamaswap-image-models
funding_url: https://github.com/mayerwin/open-webui-llamaswap-image-models
version: 1.1.0
license: MIT
description: Image-generation models routed through llama-swap, as selectable Open WebUI models. Images are stored in Open WebUI's file store so chats stay fast.
required_open_webui_version: 0.5.0
"""
# (Kept out of the YAML frontmatter above, which only allows single-line values.)
#
# Turns image-generation backends behind a single llama-swap into selectable
# models in Open WebUI. Each configured model appears in the model dropdown
# (icon-prefixed); pick one, type a prompt, get an image saved in the chat.
# Routes each model to the right backend THROUGH llama-swap, so the GPU stays
# llama-swap-coordinated (no second engine, no OOM):
#   - backend "comfyui": builds a ComfyUI workflow and POSTs it to
#       <llama-swap>/upstream/<upstream>/prompt   (FLUX.2, Qwen-Image, HiDream)
#   - backend "openai":  POSTs an OpenAI-style request to
#       <llama-swap>/upstream/<upstream>/v1/images/generations   (e.g. Ideogram)
#
# PORTABILITY: the only thing to change on another machine is the Valves --
# LLAMASWAP_URL and MODELS_JSON. MODELS_JSON maps each llama-swap image model to
# a built-in workflow template (flux2 / qwen_image / hidream / sdxl) plus its file names,
# or marks it backend "openai". No code changes for the common cases.
#
# IMAGE DELIVERY: the generated image is uploaded to Open WebUI's own file store
# and referenced by URL. Returning a base64 data URI instead (the 1.0 behaviour,
# still available via the INLINE_IMAGES valve) re-sends the whole image with
# every later message in the chat, which makes an image-heavy chat slow to load
# and painful to keep talking in.

import json
import re
import time
import base64
import asyncio

import aiohttp
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Built-in ComfyUI workflow templates (graph builders). Each takes the model's
# file names + params and returns a ComfyUI prompt graph. To support a new model
# family, add a builder here and reference it by name from MODELS_JSON.
# ---------------------------------------------------------------------------
def _wf_flux2(files, prompt, negative, w, h, steps, guidance, seed):
    return {
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": files["unet"]}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "flux2"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": negative or "blurry, low quality"}},
        "40": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "38": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["20", 0], "guidance": guidance}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["38", 0], "negative": ["21", 0], "latent_image": ["40", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["12", 0]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "owui"}},
    }


def _wf_qwen_image(files, prompt, negative, w, h, steps, guidance, seed):
    return {
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": files["unet"]}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "qwen_image"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "13": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["10", 0], "shift": 3.0}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": negative or "blurry"}},
        "40": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["13", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["40", 0], "seed": seed, "steps": steps, "cfg": guidance or 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["12", 0]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "owui"}},
    }


def _wf_hidream(files, prompt, negative, w, h, steps, guidance, seed):
    return {
        "10": {"class_type": "HiDreamO1ModelLoader", "inputs": {"model_name": files["model_name"], "precision": "bf16", "attention": "sdpa", "download_if_missing": False}},
        "20": {"class_type": "HiDreamO1Conditioning", "inputs": {"prompt": prompt, "negative_prompt": negative or "blurry"}},
        "41": {"class_type": "HiDreamO1Sampler", "inputs": {"model": ["10", 0], "conditioning": ["20", 0], "model_type": "auto", "width": w, "height": h, "steps": steps, "seed": seed, "guidance_scale": guidance, "shift": -1, "noise_scale_start": 7.5, "noise_scale_end": 7.5, "noise_clip_std": 2.5, "dev_editing_scheduler": "flow_match", "layout_bboxes": "", "preview_every": 0, "keep_image1_aspect": False, "force_offload": False, "image": "0"}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["41", 0], "filename_prefix": "owui"}},
    }


def _wf_sdxl(files, prompt, negative, w, h, steps, guidance, seed):
    # Stock SD/SDXL checkpoint graph (one .safetensors holding unet+clip+vae).
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": files["ckpt"]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative or "text, watermark"}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": guidance or 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["4", 2]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "owui"}},
    }


TEMPLATES = {"flux2": _wf_flux2, "qwen_image": _wf_qwen_image, "hidream": _wf_hidream, "sdxl": _wf_sdxl}

# Palette emoji as a module constant -- NOT inline in any f-string, because Open
# WebUI may run Python < 3.12 where a backslash inside an f-string expression is a
# SyntaxError (it compiles the function content on load).
_PALETTE = "\U0001F3A8"
_FLAGS = 'Optional inline flags: --steps 30  --size 1216x832  --seed 7  --neg "blurry, low quality".'

# EXAMPLE config (the maintainer's own llama-swap image models). Replace these with
# YOUR models by editing the MODELS_JSON valve in Open WebUI (Admin > Functions). Each
# entry: id (unique), name, icon, backend ("comfyui" | "openai"), upstream (the
# llama-swap model name to route to); for "comfyui": template + files + step/guidance
# defaults; for "openai": nothing else (it just calls /v1/images/generations).
DEFAULT_MODELS = [
    {"id": "flux-klein", "name": "FLUX.2 Klein (fastest, ~3-5 min)", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "flux2",
     "files": {"unet": "flux-2-klein-4b-Q4_K_M.gguf", "clip": "qwen_3_4b_fp4_flux2.safetensors", "vae": "flux2-vae.safetensors"},
     "steps": 20, "guidance": 3.5,
     "description": "Fast everyday image model. Describe the picture you want. " + _FLAGS},
    {"id": "flux-dev", "name": "FLUX.2 Dev (higher quality, slow ~20 min)", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "flux2",
     "files": {"unet": "flux2-dev-Q4_K_M.gguf", "clip": "mistral_3_small_flux2_fp8.safetensors", "vae": "flux2-vae.safetensors"},
     "steps": 20, "guidance": 3.5,
     "description": "Higher quality than Klein, much slower. Describe the picture you want. " + _FLAGS},
    {"id": "qwen-edit", "name": "Qwen-Image", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "qwen_image",
     "files": {"unet": "qwen-image-edit-2511-Q4_K_M.gguf", "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "vae": "qwen_image_vae.safetensors"},
     "steps": 20, "guidance": 4.0,
     "description": "Qwen image model, strong at text rendering. Describe the picture you want. " + _FLAGS},
    {"id": "hidream", "name": "HiDream-O1 (artistic, slow)", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "hidream",
     "files": {"model_name": "HiDream-O1-Image-Dev-2604-FP8"},
     "steps": 28, "guidance": 3.5,
     "description": "Artistic, painterly results, slow. Describe the picture you want. " + _FLAGS},
    {"id": "ideogram", "name": "Ideogram 4 (text in images, slow)", "icon": _PALETTE,
     "backend": "openai", "upstream": "ideogram-4",
     "description": "Best at rendering legible text/logos in images, slow. Describe the picture you want."},
]

_SUGGESTIONS = [
    {"content": "a red fox in a snowy pine forest at dawn, cinematic"},
    {"content": "a cozy cabin interior, warm lighting, photorealistic"},
    {"content": "a watercolor painting of a coastal village at sunset"},
    {"content": "a studio product photo of a vintage camera on white marble"},
]


class Pipe:
    class Valves(BaseModel):
        LLAMASWAP_URL: str = Field(
            default="http://127.0.0.1:8080",
            description="Base URL of llama-swap (it fronts ComfyUI + the image backends).")
        MODELS_JSON: str = Field(
            default=json.dumps(DEFAULT_MODELS, indent=1),
            description="JSON list of image models to expose. Each maps a llama-swap model to a "
                        "workflow template (flux2/qwen_image/hidream/sdxl) + files, or backend 'openai'.")
        WEBUI_URL: str = Field(
            default="",
            description="Base URL of this Open WebUI instance, used to upload the generated image "
                        "to its file store. Leave empty to auto-detect it from the incoming request.")
        INLINE_IMAGES: bool = Field(
            default=False,
            description="Return the image as an inline base64 data URI instead of a file-store link. "
                        "Needs no auth, but bloats the chat and slows it down badly over time.")
        TIMEOUT_S: int = Field(default=2400, description="Max seconds to wait for one image (heavy models are slow).")

    class UserValves(BaseModel):
        width: int = Field(default=1024, description="Default image width (override per-prompt with --size WxH).")
        height: int = Field(default=1024, description="Default image height.")
        steps: int = Field(default=0, description="Sampling steps (0 = the model's default; override with --steps N).")
        seed: int = Field(default=-1, description="Seed (-1 = random; override with --seed N).")
        negative: str = Field(default="", description="Default negative prompt (override with --neg ...).")

    def __init__(self):
        self.valves = self.Valves()

    def _models(self):
        try:
            data = json.loads(self.valves.MODELS_JSON)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [m for m in data if isinstance(m, dict) and m.get("id")]

    def pipes(self):
        # Manifold: each configured model becomes a selectable entry. The description
        # surfaces on the model's splash screen and advertises the inline flags.
        out = []
        for m in self._models():
            out.append({
                "id": m["id"],
                "name": f'{m.get("icon", _PALETTE)} {m.get("name", m["id"])}',
                "description": m.get("description", "Image generator. Describe the picture you want."),
                "suggestion_prompts": _SUGGESTIONS,
            })
        return out

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _collect_images(outputs):
        # ComfyUI reports a node's outputs in more than one shape depending on its
        # version and on whether asset enrichment (--enable-assets) is on: a plain
        # {"images": [...]}, the same nested under "ui", or a list of such dicts.
        # Accept all of them, otherwise a perfectly good render looks like "no image".
        imgs = []
        for n in outputs.values():
            if isinstance(n, list):
                for item in n:
                    if isinstance(item, dict):
                        imgs.extend(item.get("images") or [])
                continue
            if not isinstance(n, dict):
                continue
            imgs.extend(n.get("images") or [])
            ui = n.get("ui")
            if isinstance(ui, dict):
                imgs.extend(ui.get("images") or [])
        return [im for im in imgs if isinstance(im, dict) and im.get("filename")]

    @staticmethod
    def _last_prompt(body):
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
                return (c or "").strip()
        return ""

    @staticmethod
    def _parse_params(prompt, uv, cfg):
        # Inline overrides, e.g. 'a cat --steps 30 --size 1024x1024 --seed 7 --neg ugly'.
        w, h = uv.width, uv.height
        steps = uv.steps or cfg.get("steps", 20)
        seed = uv.seed if uv.seed is not None and uv.seed >= 0 else int(time.time() * 1000) % 2_000_000_000
        negative = uv.negative
        m = re.search(r"--size\s+(\d+)[xX](\d+)", prompt)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
        m = re.search(r"--steps\s+(\d+)", prompt)
        if m:
            steps = int(m.group(1))
        m = re.search(r"--seed\s+(\d+)", prompt)
        if m:
            seed = int(m.group(1))
        m = re.search(r'--neg\s+(?:"([^"]*)"|(\S.*?))\s*$', prompt)
        if m:
            negative = m.group(1) if m.group(1) is not None else m.group(2)
        # strip recognised flags from the prompt text so they aren't fed to the model
        clean = re.sub(r'\s*--(?:size\s+\d+[xX]\d+|steps\s+\d+|seed\s+\d+|neg\s+(?:"[^"]*"|\S.*$))', "", prompt).strip()
        # clamp size: multiples of 8, sane bounds
        w = max(256, min(2048, (max(1, w) // 8) * 8))
        h = max(256, min(2048, (max(1, h) // 8) * 8))
        steps = max(1, min(100, steps))
        return clean or prompt.strip(), w, h, steps, seed, negative

    async def _emit(self, emitter, desc, done=False):
        if emitter:
            await emitter({"type": "status", "data": {"description": desc, "done": done}})

    # ---- backends ----------------------------------------------------------
    async def _comfyui(self, s, base, cfg, prompt, neg, w, h, steps, seed, emitter):
        builder = TEMPLATES.get(cfg.get("template"))
        if not builder:
            return None, f"unknown template '{cfg.get('template')}'"
        graph = builder(cfg["files"], prompt, neg, w, h, steps, cfg.get("guidance", 3.5), seed)
        up = f"{base}/upstream/{cfg['upstream']}"
        async with s.post(f"{up}/prompt", json={"prompt": graph}) as r:
            if r.status != 200:
                return None, f"ComfyUI /prompt {r.status}: {(await r.text())[:200]}"
            pid = (await r.json()).get("prompt_id")
        if not pid:
            return None, "ComfyUI did not return a prompt_id"
        t0 = time.time()
        while time.time() - t0 < self.valves.TIMEOUT_S:
            await asyncio.sleep(3)
            try:
                async with s.get(f"{up}/history/{pid}") as hr:
                    hist = await hr.json() if hr.status == 200 else {}
            except Exception:
                # keep the timer moving so a transient poll failure does not look like a hang
                await self._emit(emitter, f"rendering on the GPU... {int(time.time() - t0)}s")
                continue
            if pid in hist:
                outs = hist[pid].get("outputs", {})
                imgs = self._collect_images(outs)
                if not imgs:
                    return None, str(hist[pid].get("status", "no image produced"))[:200]
                im = imgs[0]
                from urllib.parse import urlencode
                q = urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                async with s.get(f"{up}/view?{q}") as vr:
                    if vr.status != 200:
                        return None, f"ComfyUI /view {vr.status} for {im['filename']}"
                    raw = await vr.read()
                if not raw:
                    return None, "ComfyUI returned an empty image"
                return raw, None
            await self._emit(emitter, f"rendering on the GPU... {int(time.time() - t0)}s")
        # timed out: best-effort cancel so the next request isn't stuck behind it
        try:
            await s.post(f"{up}/interrupt")
        except Exception:
            pass
        return None, f"timed out after {self.valves.TIMEOUT_S}s"

    async def _openai(self, s, base, cfg, prompt, w, h, seed):
        url = f"{base}/upstream/{cfg['upstream']}/v1/images/generations"
        payload = {"model": cfg["upstream"], "prompt": prompt, "size": f"{w}x{h}", "n": 1}
        if seed is not None and seed >= 0:
            payload["seed"] = seed
        async with s.post(url, json=payload) as r:
            if r.status != 200:
                return None, f"images/generations {r.status}: {(await r.text())[:200]}"
            arr = (await r.json()).get("data") or []
        if not arr:
            return None, "no image in response"
        data = arr[0]
        if data.get("b64_json"):
            try:
                return base64.b64decode(data["b64_json"]), None
            except Exception as e:
                return None, f"bad base64 in response: {e}"
        if data.get("url"):
            async with s.get(data["url"]) as ir:
                if ir.status != 200:
                    return None, f"fetch image url {ir.status}"
                return await ir.read(), None
        return None, "no image in response"

    # ---- delivery ----------------------------------------------------------
    @staticmethod
    def _auth_headers(request):
        # Open WebUI's file API needs the caller's credentials; reuse whatever the
        # incoming request carried (an Authorization header, or the session cookie).
        headers = {}
        if request is None:
            return headers
        try:
            auth = request.headers.get("authorization")
        except Exception:
            auth = None
        try:
            token = request.cookies.get("token")
        except Exception:
            token = None
        if auth:
            headers["Authorization"] = auth
        if token:
            headers["Cookie"] = f"token={token}"
            headers.setdefault("Authorization", f"Bearer {token}")
        return headers

    async def _upload_to_webui(self, s, raw, prompt, request, metadata):
        base = (self.valves.WEBUI_URL or "").rstrip("/")
        if not base and request is not None:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = ""
        if not base:
            return None, "WEBUI_URL is empty and could not be auto-detected"
        meta = metadata if isinstance(metadata, dict) else {}
        try:
            form = aiohttp.FormData()
            form.add_field("file", raw, filename=f"image-{int(time.time())}.png", content_type="image/png")
            form.add_field("metadata", json.dumps({
                "prompt": prompt[:500],
                "chat_id": meta.get("chat_id", ""),
                "message_id": meta.get("message_id", ""),
            }))
            # Two details make this work, and getting either wrong is what made the
            # file store look unusable before:
            #   - the trailing slash: the route is POST /api/v1/files/ , and without
            #     it the request lands on the GET listing and comes back 405;
            #   - process=false: skips the document-extraction pipeline, which is
            #     what hangs on an image.
            url = f"{base}/api/v1/files/?process=false"
            async with s.post(url, data=form, headers=self._auth_headers(request)) as r:
                if r.status != 200:
                    return None, f"{r.status}: {(await r.text())[:200]}"
                fid = (await r.json()).get("id")
            if not fid:
                return None, "upload returned no file id"
            # Relative on purpose: the browser resolves it against the Open WebUI
            # origin it is already on, so this works from any network.
            return f"/api/v1/files/{fid}/content", None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    async def _deliver(self, s, raw, prompt, request, metadata):
        # Returns (image_src, warning). File store first, data URI as a fallback so
        # a slow render is never thrown away just because the upload failed.
        warn = None
        if not self.valves.INLINE_IMAGES:
            url, uerr = await self._upload_to_webui(s, raw, prompt, request, metadata)
            if url:
                return url, None
            warn = f"file store upload failed ({uerr}), sent inline instead"
        return "data:image/png;base64," + base64.b64encode(raw).decode(), warn

    @staticmethod
    def _is_task(body, task, metadata):
        # Open WebUI runs background tasks (title / tags / follow-up / query / autocomplete
        # generation) against the *currently selected* model when no dedicated task model is
        # configured. For an image model that renders a throwaway image per task: slow, it
        # keeps the GPU busy long after the real image, and the chat keeps "spinning". So we
        # detect a task call and skip it. The signal is exposed differently across Open WebUI
        # versions, so check all of them.
        if task:
            return True
        for src in (metadata, (body or {}).get("metadata") if isinstance(body, dict) else None):
            if isinstance(src, dict) and src.get("task"):
                return True
        return False

    # ---- entrypoint --------------------------------------------------------
    async def pipe(self, body, __user__=None, __event_emitter__=None, __request__=None,
                   __task__=None, __metadata__=None):
        if self._is_task(body, __task__, __metadata__):
            # Not a user image request; return nothing so Open WebUI falls back gracefully
            # (the chat title defaults to the prompt, no extra render is queued).
            return ""
        mid = (body.get("model") or "").split(".", 1)[-1]
        cfg = next((m for m in self._models() if m["id"] == mid), None)
        if not cfg:
            return f"Unknown image model '{mid}'. Check the pipe's MODELS_JSON valve."
        prompt = self._last_prompt(body)
        if not prompt:
            return "Type a description of the image you want."
        uv = __user__.get("valves") if isinstance(__user__, dict) else None
        if not isinstance(uv, self.UserValves):
            uv = self.UserValves()
        prompt, w, h, steps, seed, neg = self._parse_params(prompt, uv, cfg)

        await self._emit(__event_emitter__, f"Loading {cfg.get('name', mid)} via llama-swap...")
        t0 = time.time()
        raw = err = src = warn = None
        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_S + 120)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                if cfg.get("backend") == "openai":
                    raw, err = await self._openai(s, self.valves.LLAMASWAP_URL, cfg, prompt, w, h, seed)
                else:
                    raw, err = await self._comfyui(s, self.valves.LLAMASWAP_URL, cfg, prompt, neg, w, h, steps, seed, __event_emitter__)
                if raw and not err:
                    src, warn = await self._deliver(s, raw, prompt, __request__, __metadata__)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if err or not src:
            await self._emit(__event_emitter__, f"Failed: {err}", done=True)
            return f"⚠️ Image generation failed: {err or 'no image'}"
        status = f"Generated in {int(time.time() - t0)}s ({steps} steps, {w}x{h})"
        await self._emit(__event_emitter__, f"{status} -- {warn}" if warn else status, done=True)
        # src is normally an Open WebUI file-store URL; the chat stores the link, not
        # the pixels, so a long image chat stays light. It falls back to a data URI.
        # Square brackets / newlines in the prompt would break the markdown, so the
        # alt text is stripped of them rather than trusted verbatim.
        alt = re.sub(r"[\[\]\r\n]+", " ", prompt)[:60].strip()
        return f"![{alt}]({src})"
