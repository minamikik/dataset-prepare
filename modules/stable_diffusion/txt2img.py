import requests
import base64
import io
import json
from PIL import Image, PngImagePlugin
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )

def txt2img(host, prompt="", negative_prompt="", width=512, height=512):
    try:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": 40,
            "width": width,
            "height": height,
            "batch_size": 1
        }
        response = requests.post(
            url=f'http://{host}/sdapi/v1/txt2img',
            data = json.dumps(payload),
            timeout = (5.0, 300.0)
        )
        r = response.json()
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
            png_payload = {"image": "data:image/png;base64," + i}
            response2 = requests.post(
                url=f'http://{host}/sdapi/v1/png-info',
                json=png_payload
            )
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", response2.json().get("info"))

        return image
    except Exception as e:
        print(f'{host}: txt2img : {e}')
        return None
