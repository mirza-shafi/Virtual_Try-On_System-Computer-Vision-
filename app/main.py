import os
import logging
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image, ImageDraw, ImageFilter

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
    logger.warning("HF_TOKEN not found. Inpainting model download might fail.")

from app.model import VirtualTryOnModel

try:
    try_on_model = VirtualTryOnModel()
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    try_on_model = None


def create_masked_overlay(original: Image.Image, mask: Image.Image) -> Image.Image:
    """Creates a red-tinted overlay to visually highlight the segmented region."""
    original_rgb = original.convert("RGBA").resize((512, 512))
    mask_resized = mask.resize((512, 512)).convert("L")

    overlay = Image.new("RGBA", original_rgb.size, (0, 0, 0, 0))
    red_layer = Image.new("RGBA", original_rgb.size, (255, 60, 60, 140))
    overlay.paste(red_layer, mask=mask_resized)

    blended = Image.alpha_composite(original_rgb, overlay)

    # Draw a subtle border around the mask
    draw = ImageDraw.Draw(blended)
    mask_np = np.array(mask_resized)
    ys, xs = np.where(mask_np > 128)
    if len(xs) > 0 and len(ys) > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        draw.rectangle([x0, y0, x1, y1], outline=(255, 60, 60, 220), width=3)

    return blended.convert("RGB")


def process_try_on(image: Image.Image, region: str, prompt: str, progress=gr.Progress()):
    if try_on_model is None:
        return None, None, None, None, "❌ Models failed to load. Check startup logs."
    if image is None:
        return None, None, None, None, "⚠️ Please upload a photo first."
    if not prompt.strip():
        return None, None, None, None, "⚠️ Please enter a clothing description."

    logger.info(f"Processing: region={region}, prompt={prompt}")

    # ── Step 1: Show original ──────────────────────────────────────────────────
    progress(0.1, desc="Step 1 of 3 · Preparing image…")
    original_display = image.convert("RGB")

    # ── Step 2: Segmentation ───────────────────────────────────────────────────
    progress(0.3, desc="Step 2 of 3 · Segmenting clothing region…")
    try:
        mask = try_on_model.get_clothing_mask(image, region)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return original_display, None, None, None, f"❌ Segmentation Error: {e}"

    mask_display = mask.convert("RGB")  # grayscale mask as image

    # ── Step 2b: Masked overlay preview ───────────────────────────────────────
    overlay_display = create_masked_overlay(image, mask)

    # ── Step 3: Inpainting ────────────────────────────────────────────────────
    progress(0.6, desc="Step 3 of 3 · Generating new clothing with AI…")
    full_prompt = f"a person wearing {prompt}, photorealistic, high quality, detailed clothing, studio lighting"
    try:
        result = try_on_model.inpaint_with_hf(image, mask, full_prompt)
        status = f"✅ Done!  Region: {region}  ·  Prompt: '{prompt}'"
        logger.info("Try-on completed successfully.")
    except Exception as e:
        logger.error(f"Inpainting failed: {e}")
        return original_display, mask_display, overlay_display, None, f"❌ Inpainting Error: {e}"

    progress(1.0, desc="Complete!")
    return original_display, mask_display, overlay_display, result, status


# ── Custom CSS ─────────────────────────────────────────────────────────────────
custom_css = """
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Root & body ---- */
:root {
    --brand-primary: #6C63FF;
    --brand-secondary: #48D8B4;
    --accent:         #FF6B6B;
    --bg-page:        #F4F6FB;
    --bg-card:        #FFFFFF;
    --border:         #E2E8F0;
    --text-primary:   #1A202C;
    --text-muted:     #718096;
    --radius:         16px;
    --shadow:         0 4px 24px rgba(108,99,255,0.10);
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-page) !important;
    color: var(--text-primary) !important;
}

/* ---- Hero header ---- */
#hero-header {
    text-align: center;
    padding: 14px 20px 6px;
}
#hero-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 3px;
}
#hero-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
}

/* ---- Pipeline steps row ---- */
#pipeline-steps {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    padding: 6px 20px 10px;
    flex-wrap: wrap;
}
.step-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    transition: all 0.2s ease;
    white-space: nowrap;
}
.step-pill .num {
    width: 16px; height: 16px;
    border-radius: 50%;
    background: var(--border);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700;
}
.step-pill.active {
    border-color: var(--brand-primary);
    color: var(--brand-primary);
    box-shadow: 0 2px 8px rgba(108,99,255,0.15);
}
.step-pill.active .num { background: var(--brand-primary); }
.step-arrow {
    font-size: 0.85rem;
    color: var(--border);
    padding: 0 2px;
}

/* ---- Cards ---- */
.panel-card {
    background: var(--bg-card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    box-shadow: var(--shadow);
}

/* ---- Labels ---- */
label.svelte-1b6s6s, .label-wrap span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ---- Inputs (text only, not radio/checkbox) ---- */
input[type=text], input[type=email], input[type=search],
textarea, .input-text, select {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: #FAFBFF !important;
    color: #1A202C !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--brand-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.12) !important;
}
/* Gradio inner block overrides */
.block textarea, .block input[type="text"], .block input[type="email"] {
    color: #1A202C !important;
    background: #FAFBFF !important;
}
::placeholder { color: #A0AEC0 !important; opacity: 1 !important; }

/* ---- Generate button ---- */
#generate-btn {
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
    color: #fff !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 9px 18px !important;
    border: none !important;
    width: 100% !important;
    margin-top: 6px !important;
    box-shadow: 0 3px 12px rgba(108,99,255,0.28) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: pointer !important;
}
#generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.38) !important;
}
#generate-btn:active { transform: translateY(0) !important; }

/* ---- Image panels ---- */
.image-panel-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.image-panel-label .badge {
    display: inline-block;
    background: var(--brand-primary);
    color: #fff;
    border-radius: 99px;
    padding: 2px 9px;
    font-size: 0.68rem;
    font-weight: 700;
}

/* ---- Status box ---- */
#status-box textarea {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    background: #F0FDF8 !important;
    border-color: #A7F3D0 !important;
    border-radius: 10px !important;
    color: #065F46 !important;
}

/* ---- Tabs ---- */
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    color: var(--brand-primary) !important;
    border-bottom-color: var(--brand-primary) !important;
}

/* ---- Radio buttons ---- */
.wrap label {
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #1A202C !important;
}
/* Make radio inputs visible and styled */
input[type=radio] {
    appearance: auto !important;
    -webkit-appearance: radio !important;
    width: 15px !important;
    height: 15px !important;
    border: 2px solid var(--brand-primary) !important;
    background: #fff !important;
    accent-color: var(--brand-primary) !important;
    cursor: pointer !important;
    border-radius: 50% !important;
    margin-right: 5px !important;
}
/* Gradio radio group container */
.gradio-radio label, .gradio-radio span {
    color: #1A202C !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────
HERO_HTML = """
<div id="hero-header">
  <h1 id="hero-title">✦ Virtual Try-On Studio</h1>
  <p id="hero-sub">Upload a photo · pick a region · describe the look · let AI do the rest</p>
</div>
"""

PIPELINE_HTML = """
<div id="pipeline-steps">
  <div class="step-pill active"><span class="num">1</span> Upload Photo</div>
  <span class="step-arrow">→</span>
  <div class="step-pill active"><span class="num">2</span> Detect Region</div>
  <span class="step-arrow">→</span>
  <div class="step-pill active"><span class="num">3</span> Generate Clothing</div>
  <span class="step-arrow">→</span>
  <div class="step-pill active"><span class="num">4</span> Final Result</div>
</div>
"""

# ── Build Gradio UI ────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Virtual Try-On Studio",
) as demo:

    gr.HTML(HERO_HTML)
    gr.HTML(PIPELINE_HTML)

    with gr.Row(equal_height=False):

        # ── LEFT: Controls ─────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240):
            with gr.Group(elem_classes="panel-card"):
                gr.Markdown("**📸 Photo**")
                input_image = gr.Image(
                    type="pil",
                    label="Upload a portrait",
                    height=220,
                    show_label=False,
                )
                gr.Markdown("**🎯 Settings**")
                region = gr.Radio(
                    ["Upper", "Lower"],
                    label="Body Region",
                    value="Upper",
                )
                prompt = gr.Textbox(
                    label="Clothing Description",
                    placeholder="e.g. white linen shirt…",
                    lines=2,
                )
                btn = gr.Button(
                    "✨ Generate Try-On",
                    variant="primary",
                    elem_id="generate-btn",
                )

            status = gr.Textbox(
                label="Status",
                interactive=False,
                elem_id="status-box",
                lines=1,
            )

        # ── RIGHT: Visual Pipeline Output ──────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("**🔬 Pipeline**")

            with gr.Row():
                with gr.Column(min_width=100):
                    gr.HTML('<div class="image-panel-label"><span class="badge">1</span> Original</div>')
                    step1_img = gr.Image(label="", height=190, show_label=False)

                with gr.Column(min_width=100):
                    gr.HTML('<div class="image-panel-label"><span class="badge">2</span> Mask</div>')
                    step2_img = gr.Image(label="", height=190, show_label=False)

                with gr.Column(min_width=100):
                    gr.HTML('<div class="image-panel-label"><span class="badge">3</span> Overlay</div>')
                    step3_img = gr.Image(label="", height=190, show_label=False)

                with gr.Column(min_width=100):
                    gr.HTML('<div class="image-panel-label"><span class="badge" style="background:#48D8B4;color:#065F46;">✓</span> Result</div>')
                    step4_img = gr.Image(label="", height=190, show_label=False)

    btn.click(
        fn=process_try_on,
        inputs=[input_image, region, prompt],
        outputs=[step1_img, step2_img, step3_img, step4_img, status],
    )

if __name__ == "__main__":
    logger.info("Starting Gradio app…")
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="teal",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=custom_css,
    )
