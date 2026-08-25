"""
title: Image Generation (llama-swap)
author: mayerwin
author_url: https://github.com/mayerwin/open-webui-llamaswap-image-models
funding_url: https://github.com/mayerwin/open-webui-llamaswap-image-models
version: 1.2.0
license: MIT
description: Image-generation models routed through llama-swap, as selectable Open WebUI models. Supports text-to-image, image-to-image, dynamic ComfyUI workflow templates, and file-store uploads.
required_open_webui_version: 0.5.0
"""
# (Kept out of the YAML frontmatter above, which only allows single-line values.)
#
# Turns image-generation backends behind a single llama-swap into selectable
# models in Open WebUI. Each configured model appears in the model dropdown
# (icon-prefixed); pick one, type a prompt (and optionally attach an image for
# img2img), get an image saved in the chat.
# Routes each model to the right backend THROUGH llama-swap, so the GPU stays
# llama-swap-coordinated (no second engine, no OOM):
#   - backend "comfyui": builds a ComfyUI workflow (from built-in or custom dynamic
#       JSON template) and POSTs it to <llama-swap>/upstream/<upstream>/prompt
#   - backend "openai":  POSTs an OpenAI-style request to
#       <llama-swap>/upstream/<upstream>/v1/images/generations   (e.g. Ideogram)
#
# PORTABILITY: the only thing to change on another machine is the Valves --
# LLAMASWAP_URL and MODELS_JSON. MODELS_JSON maps each llama-swap image model to
# a built-in workflow template (flux2 / qwen_image / hidream / sdxl / sdxl_img2img /
# flux2_img2img) or an inline custom ComfyUI graph with {{prompt}} placeholders,
# or marks it backend "openai". No code changes for custom workflows.
#
# IMAGE DELIVERY: the generated image/video is uploaded to Open WebUI's own file
# store and referenced by URL. Returning a base64 data URI instead (the 1.0 behaviour,
# still available via the INLINE_IMAGES valve) re-sends the whole image with
# every later message in the chat, which makes an image-heavy chat slow to load.

import json
import re
import time
import base64
import asyncio
from urllib.parse import urlencode, urljoin

import aiohttp
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Built-in ComfyUI workflow templates (graph builders). Each takes the model's
# file names + params and returns a ComfyUI prompt graph.
# ---------------------------------------------------------------------------
def _wf_flux2(files, prompt, negative, w, h, steps, guidance, seed, denoise=1.0, input_image=None):
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


def _wf_flux2_img2img(files, prompt, negative, w, h, steps, guidance, seed, denoise=0.7, input_image="input.png"):
    return {
        "10": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": files["unet"]}},
        "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "flux2"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": input_image or "input.png"}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["12", 0]}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": prompt}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": negative or "blurry, low quality"}},
        "38": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["20", 0], "guidance": guidance}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["38", 0], "negative": ["21", 0], "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["12", 0]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "owui"}},
    }


def _wf_qwen_image(files, prompt, negative, w, h, steps, guidance, seed, denoise=1.0, input_image=None):
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


def _wf_hidream(files, prompt, negative, w, h, steps, guidance, seed, denoise=1.0, input_image=None):
    return {
        "10": {"class_type": "HiDreamO1ModelLoader", "inputs": {"model_name": files["model_name"], "precision": "bf16", "attention": "sdpa", "download_if_missing": False}},
        "20": {"class_type": "HiDreamO1Conditioning", "inputs": {"prompt": prompt, "negative_prompt": negative or "blurry"}},
        "41": {"class_type": "HiDreamO1Sampler", "inputs": {"model": ["10", 0], "conditioning": ["20", 0], "model_type": "auto", "width": w, "height": h, "steps": steps, "seed": seed, "guidance_scale": guidance, "shift": -1, "noise_scale_start": 7.5, "noise_scale_end": 7.5, "noise_clip_std": 2.5, "dev_editing_scheduler": "flow_match", "layout_bboxes": "", "preview_every": 0, "keep_image1_aspect": False, "force_offload": False, "image": "0"}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["41", 0], "filename_prefix": "owui"}},
    }


def _wf_sdxl(files, prompt, negative, w, h, steps, guidance, seed, denoise=1.0, input_image=None):
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


def _wf_sdxl_img2img(files, prompt, negative, w, h, steps, guidance, seed, denoise=0.7, input_image="input.png"):
    # Stock SD/SDXL Image-to-Image graph.
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": files["ckpt"]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": input_image or "input.png"}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["4", 2]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative or "text, watermark, blurry"}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": guidance or 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": denoise}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["4", 2]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "owui"}},
    }


TEMPLATES = {
    "flux2": _wf_flux2,
    "flux2_img2img": _wf_flux2_img2img,
    "qwen_image": _wf_qwen_image,
    "hidream": _wf_hidream,
    "sdxl": _wf_sdxl,
    "sdxl_img2img": _wf_sdxl_img2img,
}

# Palette emoji as a module constant -- NOT inline in any f-string, because Open
# WebUI may run Python < 3.12 where a backslash inside an f-string expression is a
# SyntaxError (it compiles the function content on load).
_PALETTE = "\U0001F3A8"
_FLAGS = 'Optional inline flags: --steps 30  --size 1216x832  --seed 7  --denoise 0.7  --neg "blurry, low quality".'

DEFAULT_MODELS = [
    {"id": "flux-klein", "name": "FLUX.2 Klein (fastest, ~3-5 min)", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "flux2",
     "files": {"unet": "flux-2-klein-4b-Q4_K_M.gguf", "clip": "qwen_3_4b_fp4_flux2.safetensors", "vae": "flux2-vae.safetensors"},
     "steps": 20, "guidance": 3.5,
     "description": "Fast everyday image model. Describe the picture you want (or attach an image). " + _FLAGS},
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
    {"id": "sdxl", "name": "SDXL / SD Checkpoint", "icon": _PALETTE,
     "backend": "comfyui", "upstream": "comfyui", "template": "sdxl",
     "files": {"ckpt": "sd_xl_base_1.0.safetensors"},
     "steps": 20, "guidance": 7.0,
     "description": "Standard SD/SDXL single checkpoint model. Describe the picture you want. " + _FLAGS},
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
                        "workflow template (flux2/qwen_image/hidream/sdxl/etc.) or inline custom workflow, or backend 'openai'.")
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
        denoise: float = Field(default=0.7, description="Denoising strength for image-to-image (0.0 to 1.0; override with --denoise N).")
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
    def _render_template(template_obj, ctx):
        # Recursively walks a dict or list, replacing {{placeholder}} tags with ctx values.
        # If the string is an exact match for a numeric/typed placeholder, casts to the proper type.
        if isinstance(template_obj, dict):
            return {k: Pipe._render_template(v, ctx) for k, v in template_obj.items()}
        if isinstance(template_obj, list):
            return [Pipe._render_template(x, ctx) for x in template_obj]
        if not isinstance(template_obj, str):
            return template_obj

        val = template_obj.strip()
        # Direct exact substitutions with type conversion
        if val == "{{prompt}}":
            return ctx.get("prompt", "")
        if val == "{{negative}}":
            return ctx.get("negative", "")
        if val == "{{width}}":
            return int(ctx.get("width", 1024))
        if val == "{{height}}":
            return int(ctx.get("height", 1024))
        if val == "{{steps}}":
            return int(ctx.get("steps", 20))
        if val == "{{seed}}":
            return int(ctx.get("seed", 0))
        if val in ("{{guidance}}", "{{cfg}}"):
            return float(ctx.get("guidance", 3.5))
        if val == "{{denoise}}":
            return float(ctx.get("denoise", 0.7))
        if val == "{{input_image}}":
            return str(ctx.get("input_image") or "input.png")

        # Nested {{files.<key>}} exact substitution
        m_file = re.fullmatch(r"\{\{files\.([\w\.\-]+)\}\}", val)
        if m_file:
            key = m_file.group(1)
            files = ctx.get("files") or {}
            return files.get(key, "")

        # Partial substring regex replacements
        def _sub(match):
            k = match.group(1).strip()
            if k.startswith("files."):
                fkey = k[6:]
                return str((ctx.get("files") or {}).get(fkey, ""))
            return str(ctx.get(k, ""))

        return re.sub(r"\{\{([\w\.\-]+)\}\}", _sub, template_obj)

    @staticmethod
    def _collect_media(outputs):
        # Collects images or videos from ComfyUI outputs across all output formats and nesting shapes.
        items = []
        for n in outputs.values():
            target_list = []
            if isinstance(n, list):
                for item in n:
                    if isinstance(item, dict):
                        target_list.append(item)
            elif isinstance(n, dict):
                target_list.append(n)
                ui = n.get("ui")
                if isinstance(ui, dict):
                    target_list.append(ui)

            for d in target_list:
                for k in ("images", "gifs", "videos"):
                    arr = d.get(k)
                    if isinstance(arr, list):
                        items.extend(arr)

        return [m for m in items if isinstance(m, dict) and m.get("filename")]

    @staticmethod
    def _last_user_message(body):
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                return msg
        return None

    @classmethod
    def _last_prompt(cls, body):
        msg = cls._last_user_message(body)
        if not msg:
            return ""
        c = msg.get("content")
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
        return (c or "").strip()

    @classmethod
    async def _extract_input_image(cls, body, request, s, webui_url):
        # Extracts user-attached images from the last message in Open WebUI.
        msg = cls._last_user_message(body)
        if not msg:
            return None, None

        img_url = None
        c = msg.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    img_info = part.get("image_url")
                    if isinstance(img_info, dict) and img_info.get("url"):
                        img_url = img_info["url"]
                        break
                    if isinstance(img_info, str):
                        img_url = img_info
                        break

        # Also check msg.get("files") or body.get("files")
        if not img_url:
            files = msg.get("files") or body.get("files") or []
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict):
                        furl = f.get("url") or f.get("file_path") or f.get("id")
                        if furl:
                            img_url = f"/api/v1/files/{furl}/content" if not str(furl).startswith(("/", "http")) else str(furl)
                            break

        if not img_url:
            return None, None

        # 1. Base64 data URI
        if img_url.startswith("data:image/"):
            try:
                header, b64 = img_url.split(",", 1)
                ext = "png"
                if "jpeg" in header or "jpg" in header:
                    ext = "jpg"
                elif "webp" in header:
                    ext = "webp"
                return base64.b64decode(b64), f"input-{int(time.time())}.{ext}"
            except Exception:
                return None, None

        # 2. Open WebUI internal file or HTTP URL
        base = (webui_url or "").rstrip("/")
        if not base and request is not None:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = ""

        fetch_url = img_url
        if img_url.startswith("/"):
            if not base:
                return None, None
            fetch_url = base + img_url

        try:
            headers = cls._auth_headers(request)
            async with s.get(fetch_url, headers=headers) as r:
                if r.status == 200:
                    raw = await r.read()
                    filename = f"input-{int(time.time())}.png"
                    return raw, filename
        except Exception:
            pass

        return None, None

    @staticmethod
    def _parse_params(prompt, uv, cfg):
        # Inline overrides, e.g. 'a cat --steps 30 --size 1024x1024 --seed 7 --denoise 0.6 --neg ugly'.
        w, h = uv.width, uv.height
        steps = uv.steps or cfg.get("steps", 20)
        seed = uv.seed if uv.seed is not None and uv.seed >= 0 else int(time.time() * 1000) % 2_000_000_000
        denoise = getattr(uv, "denoise", 0.7)
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
        m = re.search(r"--denoise\s+(\d+(?:\.\d+)?)", prompt)
        if m:
            denoise = float(m.group(1))
        m = re.search(r'--neg\s+(?:"([^"]*)"|(\S.*?))\s*$', prompt)
        if m:
            negative = m.group(1) if m.group(1) is not None else m.group(2)

        clean = re.sub(r'\s*--(?:size\s+\d+[xX]\d+|steps\s+\d+|seed\s+\d+|denoise\s+\d+(?:\.\d+)?|neg\s+(?:"[^"]*"|\S.*$))', "", prompt).strip()
        w = max(256, min(2048, (max(1, w) // 8) * 8))
        h = max(256, min(2048, (max(1, h) // 8) * 8))
        steps = max(1, min(100, steps))
        denoise = max(0.0, min(1.0, float(denoise)))
        return clean or prompt.strip(), w, h, steps, seed, negative, denoise

    async def _emit(self, emitter, desc, done=False):
        if emitter:
            await emitter({"type": "status", "data": {"description": desc, "done": done}})

    # ---- backends ----------------------------------------------------------
    async def _upload_input_image_to_comfyui(self, s, base, upstream, img_bytes, filename):
        # Uploads input image to ComfyUI via POST /upload/image
        up = f"{base}/upstream/{upstream}"
        form = aiohttp.FormData()
        form.add_field("image", img_bytes, filename=filename, content_type="image/png")
        form.add_field("overwrite", "true")
        async with s.post(f"{up}/upload/image", data=form) as r:
            if r.status != 200:
                return None, f"ComfyUI /upload/image returned {r.status}"
            data = await r.json()
            return data.get("name") or filename, None

    async def _comfyui(self, s, base, cfg, prompt, neg, w, h, steps, seed, denoise, input_image_name, emitter):
        # Determine graph: inline custom workflow, or built-in template
        custom_wf = cfg.get("workflow") or (cfg.get("template") if isinstance(cfg.get("template"), dict) else None)
        if custom_wf:
            ctx = {
                "prompt": prompt,
                "negative": neg or "",
                "width": w,
                "height": h,
                "steps": steps,
                "seed": seed,
                "guidance": cfg.get("guidance", 3.5),
                "cfg": cfg.get("guidance", 3.5),
                "denoise": denoise,
                "input_image": input_image_name or "input.png",
                "files": cfg.get("files") or {},
            }
            graph = self._render_template(custom_wf, ctx)
        else:
            tmpl_name = cfg.get("template")
            # Auto-switch to img2img template if an image is provided and a corresponding template exists
            if input_image_name and tmpl_name in ("sdxl", "flux2"):
                tmpl_name = f"{tmpl_name}_img2img"

            builder = TEMPLATES.get(tmpl_name)
            if not builder:
                return None, None, f"unknown template '{cfg.get('template')}'"
            try:
                graph = builder(cfg.get("files") or {}, prompt, neg, w, h, steps, cfg.get("guidance", 3.5), seed, denoise=denoise, input_image=input_image_name)
            except TypeError:
                # Older builder signature fallback
                graph = builder(cfg.get("files") or {}, prompt, neg, w, h, steps, cfg.get("guidance", 3.5), seed)

        up = f"{base}/upstream/{cfg['upstream']}"
        async with s.post(f"{up}/prompt", json={"prompt": graph}) as r:
            if r.status != 200:
                return None, None, f"ComfyUI /prompt {r.status}: {(await r.text())[:200]}"
            pid = (await r.json()).get("prompt_id")
        if not pid:
            return None, None, "ComfyUI did not return a prompt_id"

        t0 = time.time()
        while time.time() - t0 < self.valves.TIMEOUT_S:
            await asyncio.sleep(3)
            try:
                async with s.get(f"{up}/history/{pid}") as hr:
                    hist = await hr.json() if hr.status == 200 else {}
            except Exception:
                await self._emit(emitter, f"rendering on the GPU... {int(time.time() - t0)}s")
                continue
            if pid in hist:
                status_obj = hist[pid].get("status", {})
                if isinstance(status_obj, dict) and status_obj.get("status_str") == "error":
                    messages = status_obj.get("messages", [])
                    err_msg = "ComfyUI execution error"
                    for m in messages:
                        if isinstance(m, list) and len(m) > 1 and m[0] == "execution_error":
                            err_msg = m[1].get("exception_message") or str(m[1])
                            break
                    return None, None, err_msg[:300]

                outs = hist[pid].get("outputs", {})
                media_items = self._collect_media(outs)
                if media_items:
                    item = media_items[0]
                    q = urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
                    async with s.get(f"{up}/view?{q}") as vr:
                        if vr.status != 200:
                            return None, None, f"ComfyUI /view {vr.status} for {item['filename']}"
                        raw = await vr.read()
                    if not raw:
                        return None, None, "ComfyUI returned empty output"
                    return raw, item["filename"], None

                if isinstance(status_obj, dict) and status_obj.get("completed") is True:
                    return None, None, "Workflow completed but produced no image/video output"
            await self._emit(emitter, f"rendering on the GPU... {int(time.time() - t0)}s")

        # timed out: best-effort cancel
        try:
            await s.post(f"{up}/interrupt")
        except Exception:
            pass
        return None, None, f"timed out after {self.valves.TIMEOUT_S}s"

    async def _openai(self, s, base, cfg, prompt, w, h, seed):
        url = f"{base}/upstream/{cfg['upstream']}/v1/images/generations"
        payload = {"model": cfg["upstream"], "prompt": prompt, "size": f"{w}x{h}", "n": 1}
        if seed is not None and seed >= 0:
            payload["seed"] = seed
        async with s.post(url, json=payload) as r:
            if r.status != 200:
                return None, None, f"images/generations {r.status}: {(await r.text())[:200]}"
            arr = (await r.json()).get("data") or []
        if not arr:
            return None, None, "no image in response"
        data = arr[0]
        if data.get("b64_json"):
            try:
                return base64.b64decode(data["b64_json"]), "output.png", None
            except Exception as e:
                return None, None, f"bad base64 in response: {e}"
        if data.get("url"):
            async with s.get(data["url"]) as ir:
                if ir.status != 200:
                    return None, None, f"fetch image url {ir.status}"
                return await ir.read(), "output.png", None
        return None, None, "no image in response"

    # ---- delivery ----------------------------------------------------------
    @staticmethod
    def _auth_headers(request):
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

    async def _upload_to_webui(self, s, raw, prompt, filename, request, metadata):
        base = (self.valves.WEBUI_URL or "").rstrip("/")
        if not base and request is not None:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = ""
        if not base:
            return None, "WEBUI_URL is empty and could not be auto-detected"

        ext = (filename or "image.png").split(".")[-1].lower()
        content_type = "image/png"
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext == "webp":
            content_type = "image/webp"
        elif ext == "mp4":
            content_type = "video/mp4"
        elif ext == "webm":
            content_type = "video/webm"
        elif ext == "gif":
            content_type = "image/gif"

        meta = metadata if isinstance(metadata, dict) else {}
        try:
            form = aiohttp.FormData()
            form.add_field("file", raw, filename=filename or f"media-{int(time.time())}.{ext}", content_type=content_type)
            form.add_field("metadata", json.dumps({
                "prompt": prompt[:500],
                "chat_id": meta.get("chat_id", ""),
                "message_id": meta.get("message_id", ""),
            }))
            url = f"{base}/api/v1/files/?process=false"
            async with s.post(url, data=form, headers=self._auth_headers(request)) as r:
                if r.status != 200:
                    return None, f"{r.status}: {(await r.text())[:200]}"
                fid = (await r.json()).get("id")
            if not fid:
                return None, "upload returned no file id"
            return f"/api/v1/files/{fid}/content", None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    async def _deliver(self, s, raw, prompt, filename, request, metadata):
        warn = None
        ext = (filename or "output.png").split(".")[-1].lower()
        is_video = ext in ("mp4", "webm")
        mime = "video/mp4" if ext == "mp4" else ("video/webm" if ext == "webm" else f"image/{ext if ext in ('png', 'jpeg', 'webp', 'gif') else 'png'}")

        if not self.valves.INLINE_IMAGES:
            url, uerr = await self._upload_to_webui(s, raw, prompt, filename, request, metadata)
            if url:
                return url, is_video, None
            warn = f"file store upload failed ({uerr}), sent inline instead"

        data_uri = f"data:{mime};base64," + base64.b64encode(raw).decode()
        return data_uri, is_video, warn

    @staticmethod
    def _is_task(body, task, metadata):
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
        prompt, w, h, steps, seed, neg, denoise = self._parse_params(prompt, uv, cfg)

        await self._emit(__event_emitter__, f"Loading {cfg.get('name', mid)} via llama-swap...")
        t0 = time.time()
        raw = filename = err = src = warn = None
        is_video = False

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_S + 120)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                # Check for attached input image for img2img / img2vid
                input_bytes, in_fname = await self._extract_input_image(body, __request__, s, self.valves.WEBUI_URL)
                input_image_name = None
                if input_bytes and cfg.get("backend") != "openai":
                    await self._emit(__event_emitter__, "Uploading reference image to ComfyUI...")
                    input_image_name, uperr = await self._upload_input_image_to_comfyui(
                        s, self.valves.LLAMASWAP_URL, cfg["upstream"], input_bytes, in_fname or "input.png")
                    if uperr:
                        await self._emit(__event_emitter__, f"Image upload warning: {uperr}")

                if cfg.get("backend") == "openai":
                    raw, filename, err = await self._openai(s, self.valves.LLAMASWAP_URL, cfg, prompt, w, h, seed)
                else:
                    raw, filename, err = await self._comfyui(
                        s, self.valves.LLAMASWAP_URL, cfg, prompt, neg, w, h, steps, seed, denoise, input_image_name, __event_emitter__)

                if raw and not err:
                    src, is_video, warn = await self._deliver(s, raw, prompt, filename, __request__, __metadata__)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        if err or not src:
            await self._emit(__event_emitter__, f"Failed: {err}", done=True)
            return f"⚠️ Image generation failed: {err or 'no media generated'}"

        status = f"Generated in {int(time.time() - t0)}s ({steps} steps, {w}x{h})"
        await self._emit(__event_emitter__, f"{status} -- {warn}" if warn else status, done=True)

        alt = re.sub(r"[\[\]\r\n]+", " ", prompt)[:60].strip()
        if is_video:
            return f'<video controls autoplay loop src="{src}" style="max-width: 100%; border-radius: 8px;"></video>'
        return f"![{alt}]({src})"
