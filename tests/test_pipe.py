import unittest
import asyncio
import json
import base64
from aiohttp import web
from llamaswap_image_models import Pipe, DEFAULT_MODELS, TEMPLATES


class TestPipe(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipe()

    def test_syntax_and_pipes_loading(self):
        pipes = self.pipe.pipes()
        self.assertTrue(len(pipes) >= 5)
        ids = [p["id"] for p in pipes]
        self.assertIn("flux-klein", ids)
        self.assertIn("sdxl", ids)
        self.assertIn("hidream", ids)

    def test_render_template_types_and_placeholders(self):
        raw_graph = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "{{files.ckpt}}"}
            },
            "2": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": "{{width}}",
                    "height": "{{height}}",
                    "batch_size": 1
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": "{{seed}}",
                    "steps": "{{steps}}",
                    "cfg": "{{guidance}}",
                    "denoise": "{{denoise}}",
                    "note": "Prompt was: {{prompt}} and neg: {{negative}}"
                }
            },
            "4": {
                "class_type": "LoadImage",
                "inputs": {"image": "{{input_image}}"}
            }
        }
        ctx = {
            "prompt": "a futuristic city",
            "negative": "low quality",
            "width": 1280,
            "height": 720,
            "steps": 30,
            "seed": 9999,
            "guidance": 4.5,
            "denoise": 0.65,
            "input_image": "my_input.png",
            "files": {"ckpt": "model_v1.safetensors"}
        }
        rendered = Pipe._render_template(raw_graph, ctx)
        
        self.assertEqual(rendered["1"]["inputs"]["ckpt_name"], "model_v1.safetensors")
        self.assertEqual(rendered["2"]["inputs"]["width"], 1280)
        self.assertIsInstance(rendered["2"]["inputs"]["width"], int)
        self.assertEqual(rendered["2"]["inputs"]["height"], 720)
        self.assertIsInstance(rendered["2"]["inputs"]["height"], int)
        self.assertEqual(rendered["3"]["inputs"]["steps"], 30)
        self.assertIsInstance(rendered["3"]["inputs"]["steps"], int)
        self.assertEqual(rendered["3"]["inputs"]["seed"], 9999)
        self.assertIsInstance(rendered["3"]["inputs"]["seed"], int)
        self.assertEqual(rendered["3"]["inputs"]["cfg"], 4.5)
        self.assertIsInstance(rendered["3"]["inputs"]["cfg"], float)
        self.assertEqual(rendered["3"]["inputs"]["denoise"], 0.65)
        self.assertIsInstance(rendered["3"]["inputs"]["denoise"], float)
        self.assertEqual(rendered["3"]["inputs"]["note"], "Prompt was: a futuristic city and neg: low quality")
        self.assertEqual(rendered["4"]["inputs"]["image"], "my_input.png")

    def test_parse_params(self):
        prompt = 'a cybernetic wolf --size 1280x768 --steps 35 --seed 12345 --denoise 0.55 --neg "ugly, artifacts"'
        uv = Pipe.UserValves()
        cfg = {"steps": 20}
        clean, w, h, steps, seed, neg, denoise = Pipe._parse_params(prompt, uv, cfg)
        
        self.assertEqual(clean, "a cybernetic wolf")
        self.assertEqual(w, 1280)
        self.assertEqual(h, 768)
        self.assertEqual(steps, 35)
        self.assertEqual(seed, 12345)
        self.assertEqual(denoise, 0.55)
        self.assertEqual(neg, "ugly, artifacts")

    def test_collect_media(self):
        # Image in outputs
        outs_image = {
            "9": {"images": [{"filename": "out_001.png", "subfolder": "", "type": "output"}]}
        }
        media = Pipe._collect_media(outs_image)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["filename"], "out_001.png")

        # Nested in UI
        outs_ui = {
            "10": {"ui": {"images": [{"filename": "out_ui.webp", "type": "output"}]}}
        }
        media = Pipe._collect_media(outs_ui)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["filename"], "out_ui.webp")

        # Video from VHS_VideoCombine
        outs_video = {
            "15": {"videos": [{"filename": "render.mp4", "type": "output"}]}
        }
        media = Pipe._collect_media(outs_video)
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0]["filename"], "render.mp4")

    def test_extract_input_image_data_uri(self):
        fake_png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake_image_bytes").decode()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "enhance this photo"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fake_png_b64}"}}
                    ]
                }
            ]
        }
        
        async def _run():
            raw, fname = await Pipe._extract_input_image(body, None, None, None)
            self.assertEqual(raw, b"\x89PNG\r\n\x1a\nfake_image_bytes")
            self.assertTrue(fname.endswith(".png"))
            
        asyncio.run(_run())


class TestMockE2E(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pipe = Pipe()
        self.pipe.valves.INLINE_IMAGES = True

    async def test_mock_comfyui_flow(self):
        app = web.Application()
        
        uploaded_images = []
        prompts_received = []

        async def handle_upload(request):
            reader = await request.multipart()
            field = await reader.next()
            data = await field.read()
            uploaded_images.append(data)
            return web.json_response({"name": "uploaded_test.png", "subfolder": "", "type": "input"})

        async def handle_prompt(request):
            data = await request.json()
            prompts_received.append(data["prompt"])
            return web.json_response({"prompt_id": "test-prompt-123"})

        async def handle_history(request):
            return web.json_response({
                "test-prompt-123": {
                    "outputs": {
                        "10": {
                            "images": [{"filename": "mock_result.png", "subfolder": "", "type": "output"}]
                        }
                    }
                }
            })

        async def handle_view(request):
            return web.Response(body=b"\x89PNG\r\n\x1a\nmock_rendered_pixels", content_type="image/png")

        app.router.add_post("/upstream/comfyui/upload/image", handle_upload)
        app.router.add_post("/upstream/comfyui/prompt", handle_prompt)
        app.router.add_get("/upstream/comfyui/history/test-prompt-123", handle_history)
        app.router.add_get("/upstream/comfyui/view", handle_view)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18888)
        await site.start()

        self.pipe.valves.LLAMASWAP_URL = "http://127.0.0.1:18888"

        try:
            # 1. Text to Image
            body = {
                "model": "flux-klein",
                "messages": [{"role": "user", "content": "a cute kitten --steps 10"}]
            }
            res = await self.pipe.pipe(body)
            self.assertTrue(res.startswith("![a cute kitten](data:image/png;base64,"))
            self.assertEqual(len(prompts_received), 1)

            # 2. Image to Image with dynamic custom workflow
            fake_b64 = base64.b64encode(b"input_image_content").decode()
            custom_model = {
                "id": "custom-img2img",
                "name": "Custom Img2Img",
                "backend": "comfyui",
                "upstream": "comfyui",
                "workflow": {
                    "1": {"class_type": "LoadImage", "inputs": {"image": "{{input_image}}"}},
                    "2": {"class_type": "KSampler", "inputs": {"denoise": "{{denoise}}", "steps": "{{steps}}"}},
                    "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "owui"}}
                }
            }
            self.pipe.valves.MODELS_JSON = json.dumps([custom_model])
            
            body_img2img = {
                "model": "custom-img2img",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "oil painting style --denoise 0.6"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fake_b64}"}}
                        ]
                    }
                ]
            }
            res2 = await self.pipe.pipe(body_img2img)
            self.assertTrue(res2.startswith("![oil painting style](data:image/png;base64,"))
            self.assertEqual(len(uploaded_images), 1)
            self.assertEqual(uploaded_images[0], b"input_image_content")
            self.assertEqual(prompts_received[1]["1"]["inputs"]["image"], "uploaded_test.png")
            self.assertEqual(prompts_received[1]["2"]["inputs"]["denoise"], 0.6)

        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
