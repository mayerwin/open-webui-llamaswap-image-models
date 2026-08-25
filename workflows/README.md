# ComfyUI Source Workflows

This directory contains the original, full GUI workflow files (`.json`) corresponding to the built-in templates used by `open-webui-llamaswap-image-models`.

You can load any of these `.json` files directly into the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) web interface (via **Load** or dragging and dropping the file into the canvas).

---

## Included Workflows

| File | Family / Architecture | Required Custom Nodes | Key Models Needed |
| :--- | :--- | :--- | :--- |
| [`sdxl.json`](sdxl.json) | SD 1.5 / SDXL / Checkpoint | *None (Stock ComfyUI)* | `models/checkpoints/sd_xl_base_1.0.safetensors` |
| [`sdxl_img2img.json`](sdxl_img2img.json) | SDXL Image-to-Image | *None (Stock ComfyUI)* | `models/checkpoints/sd_xl_base_1.0.safetensors` |
| [`flux2.json`](flux2.json) | FLUX.2 (Klein / Dev) | [`ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) | `models/unet/flux-2-klein-4b-Q4_K_M.gguf`<br>`models/clip/qwen_3_4b_fp4_flux2.safetensors`<br>`models/vae/flux2-vae.safetensors` |
| [`qwen_image.json`](qwen_image.json) | Qwen-Image / Qwen-Edit | [`ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) | `models/unet/qwen-image-edit-2511-Q4_K_M.gguf`<br>`models/clip/qwen_2.5_vl_7b_fp8_scaled.safetensors`<br>`models/vae/qwen_image_vae.safetensors` |
| [`hidream.json`](hidream.json) | HiDream-O1 | [`HiDream_O1-ComfyUI`](https://github.com/Saganaki22/HiDream_O1-ComfyUI) | `models/hidream_o1/HiDream-O1-Image-Dev-2604-FP8` |

---

## Installing Missing Custom Nodes

If you have [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:
1. Drag and drop any `.json` file above into your ComfyUI browser tab.
2. If any node appears red, open **Manager** > **Install Missing Custom Nodes**.
3. Click **Install** next to the missing packages, then restart ComfyUI.

### Manual Git Clone Commands (if not using ComfyUI-Manager)

Run these inside your `ComfyUI/custom_nodes/` directory:

```bash
# For FLUX.2 and Qwen-Image GGUF weights:
git clone https://github.com/city96/ComfyUI-GGUF.git

# For HiDream-O1:
git clone https://github.com/Saganaki22/HiDream_O1-ComfyUI.git
```

---

## Exporting Your Own Custom Workflows for Open WebUI

To turn any custom ComfyUI workflow into a dynamic model in Open WebUI:
1. In ComfyUI, go to **Settings (gear icon)** and enable **Enable Dev mode Options**.
2. Click **Save (API Format)** to export the execution graph.
3. Replace the prompt/seed/size fields with template variables:
   - `{{prompt}}` — the user prompt text
   - `{{negative}}` — the negative prompt
   - `{{width}}` / `{{height}}` — image resolution
   - `{{steps}}` — number of sampling steps
   - `{{guidance}}` / `{{cfg}}` — CFG or guidance scale
   - `{{seed}}` — randomized or specified seed
   - `{{denoise}}` — denoising strength for img2img (default 0.7)
   - `{{input_image}}` — filename of the image uploaded in chat (for img2img)
   - `{{files.<key>}}` — references to model files specified under `"files"`
4. Paste the resulting JSON into the `"workflow"` field of your `MODELS_JSON` valve entry in Open WebUI!
