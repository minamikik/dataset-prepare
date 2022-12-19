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

def interrogate(host, image, model):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        payload = {
            "image": image_base64,
            "model": model
        }
        response = requests.post(
            url=f'http://{host}/sdapi/v1/interrogate',
            data = json.dumps(payload),
            timeout = (5.0, 300.0)
        )
        r = response.json()
        prompt = r['caption']

        return prompt
    except Exception as e:
        print(f'{host}: interrogate : {e}')
        return None
