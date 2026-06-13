import torch
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline
import logging

logger = logging.getLogger(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Label map for Segformer
LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

UPPER_IDS = [4, 7]      # Upper-clothes, Dress
LOWER_IDS = [5, 6, 7]   # Skirt, Pants, Dress

class VirtualTryOnModel:
    def __init__(self):
        logger.info("Initializing Virtual Try-On Model...")
        self.device = device
        
        # Load Segmentation Model
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(self.device)
        self.seg_model.eval()
        logger.info("Segmentation model loaded.")

        # Load Inpainting Model
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        logger.info("Inpainting pipeline loaded.")

    def get_clothing_mask(self, image: Image.Image, region: str) -> Image.Image:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            logits = outputs.logits  # (1, num_labels, H, W)

        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()

        target_ids = UPPER_IDS if region == "Upper" else LOWER_IDS
        mask = np.isin(pred_seg, target_ids).astype(np.uint8) * 255

        # Dilate slightly so inpainting covers edges cleanly
        mask = binary_dilation(mask, iterations=8).astype(np.uint8) * 255

        return Image.fromarray(mask).convert("L")

    def inpaint_with_hf(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        result = self.inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]

        return result
