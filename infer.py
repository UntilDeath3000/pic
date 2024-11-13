import cv2
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import insightface
import base64
import warnings

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_faceid import draw_kps

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

warnings.filterwarnings('ignore', category=FutureWarning)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

if __name__ == "__main__":
    # Load face encoder
    app = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id 表示使用的 GPU 设备 ID，默认为 -1，表示使用 CPU

    # Infer setting
    prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
    n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

    face_image = Image.open("/content/drive/MyDrive/kaggle/FaceID/result.jpg").convert('RGB')
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    # Prepare payload for Hugging Face API
    face_kps_bytes = BytesIO()
    face_kps.save(face_kps_bytes, format='PNG')
    face_kps_base64 = base64.b64encode(face_kps_bytes.getvalue()).decode('utf-8')

    payload = {
        "inputs": prompt,
        "negative_prompt": n_prompt,
        "image_embeds": face_emb.tolist(),  # Convert numpy array to list
        "image": face_kps_base64,  # Convert PIL Image to base64 encoded string
        "num_inference_steps": 30,
        "guidance_scale": 5,
    }

    # Call Hugging Face API
    image_bytes = query(payload)

    # Convert the response to an image
    image = Image.open(BytesIO(image_bytes))

    # Save the result
    image.save('/content/drive/MyDrive/kaggle/FaceID/result1.jpg')
