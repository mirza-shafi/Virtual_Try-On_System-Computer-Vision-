import os
import logging
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check for Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    logger.info("Logging into Hugging Face Hub...")
    login(token=HF_TOKEN)
else:
    logger.warning("HF_TOKEN not found in environment. Inpainting model download might fail if it requires auth.")

# Import model after setting up environment
from app.model import VirtualTryOnModel

# Initialize the model instance lazily or eagerly. We'll do eagerly for the UI to be ready.
try:
    try_on_model = VirtualTryOnModel()
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    try_on_model = None

def process_try_on(image: Image.Image, region: str, prompt: str):
    if try_on_model is None:
        return None, None, "❌ Error: Models failed to load upon startup. Check logs."

    if image is None:
        return None, None, "Please upload an image."
    if not prompt.strip():
        return None, None, "Please enter a prompt (e.g. 'black leather jacket')."

    logger.info(f"Processing try-on. Region: {region}, Prompt: {prompt}")

    # Step 1: Segmentation
    try:
        mask = try_on_model.get_clothing_mask(image, region)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return None, None, f"❌ Segmentation Error: {str(e)}"

    # Step 2: Inpainting
    full_prompt = f"a person wearing {prompt}, photorealistic, high quality, detailed clothing"
    try:
        result = try_on_model.inpaint_with_hf(image, mask, full_prompt)
        status = f"✅ Success! Region: {region}, Prompt: '{prompt}'"
        logger.info("Try-on successful.")
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")
        return mask, None, f"❌ Inpainting Error: {str(e)}"

    return mask, result, status

# Custom CSS for a premium look
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.title-text {
    text-align: center;
    font-weight: 800;
    font-size: 2.5rem;
    background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.subtitle-text {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
"""

# Define Gradio UI with a premium theme
with gr.Blocks(title="Virtual Try-On System", theme=gr.themes.Soft(primary_hue="teal", secondary_hue="rose"), css=custom_css) as demo:
    gr.HTML("""
        <div class="title-text">Virtual Try-On Studio</div>
        <div class="subtitle-text">Transform your wardrobe with AI. Upload a photo, select a region, and describe your new look.</div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                input_image = gr.Image(type="pil", label="Your Photo", height=400)
                region = gr.Radio(["Upper", "Lower"], label="Body Region to Change", value="Upper", inline=True)
                prompt = gr.Textbox(
                    label="Clothing Description", 
                    placeholder="e.g. A stylish black leather jacket...",
                    lines=2
                )
                btn = gr.Button("✨ Generate Magic ✨", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Final Result"):
                    result_output = gr.Image(label="Your New Look", height=500)
                with gr.TabItem("Segmentation Mask"):
                    mask_output = gr.Image(label="AI Detection Mask", height=500)
            
            status = gr.Textbox(label="Status updates", interactive=False)

    btn.click(
        fn=process_try_on,
        inputs=[input_image, region, prompt],
        outputs=[mask_output, result_output, status]
    )

if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

