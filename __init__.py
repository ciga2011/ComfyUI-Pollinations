import os
import requests
from PIL import Image
import io
import torch
import numpy as np
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar

class PollinationsNode:
    def __init__(self):
        self.device = get_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a futuristic cityscape"}),
                "model": (["flux", "flux-realism", "flux-anime", "flux-3d", "any-dark", "flux-pro", "turbo"], {"default": "flux"}),
                "width": ("INT", {"default": 900, "min": 1, "max": 2048}),
                "height": ("INT", {"default": 600, "min": 1, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "nologo": ("BOOLEAN", {"default": True}),
                "enhance": ("BOOLEAN", {"default": True}),
                "private": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Pollinations"

    def generate_image(self, prompt, model, width, height, seed, nologo=True, enhance=True, private=True):
        # Build the API URL
        api_url = f"https://image.pollinations.ai/prompt/{prompt}?model={model}&width={width}&height={height}&seed={seed}"
        if nologo:
            api_url += "&nologo=1"
        if enhance:
            api_url += "&enhance=1"
        if private:
            api_url += "&private=1"

        # Fetch the image from Pollinations API
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception(f"Failed to generate image: {response.status_code} - {response.text}")

        # Load the image into a PIL Image
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Convert PIL Image to a torch tensor
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        return (image_tensor,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "PollinationsNode": PollinationsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PollinationsNode": "Pollinations Free Image Generator",
}