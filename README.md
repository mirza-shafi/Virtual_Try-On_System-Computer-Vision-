# ✦ Virtual Try-On Studio

A production-grade **Virtual Try-On system** powered by semantic segmentation and Stable Diffusion inpainting. Upload a photo, select a body region, describe the clothing — and let AI generate a photorealistic new outfit.

---

## ✨ Features

- **🔬 Visual Pipeline UI** — See every step of the process: original photo → segmentation mask → region overlay → final result, all in one compact dashboard.
- **👗 Semantic Segmentation** — Accurately detects upper or lower body garments using `Segformer B2`.
- **🎨 AI Inpainting** — Replaces the masked clothing region with AI-generated designs via `Stable Diffusion Inpainting`.
- **🖥️ Clean White UI** — Premium, compact Gradio interface with live progress updates.
- **🐳 Dockerized** — Ready for deployment with a single Docker command.

---

## 🗂️ Project Structure

```text
Virtual_Try-On_System/
├── app/
│   ├── __init__.py
│   ├── main.py        # Gradio UI & pipeline orchestration
│   ├── model.py       # Segformer + Stable Diffusion logic
│   └── utils.py       # Helper utilities (image encoding, etc.)
├── Dockerfile         # Docker image configuration
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.10+**
- A [Hugging Face token](https://huggingface.co/settings/tokens) (required for model access)

### Local Setup

1. **Create & activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your Hugging Face token:
   ```
   HF_TOKEN=your_token_here
   ```

4. **Run the application:**
   ```bash
   # Use the venv Python directly (recommended on macOS)
   venv/bin/python -m app.main

   # Or if 'python3' is on your PATH:
   python3 -m app.main
   ```

   The app will start at **[http://127.0.0.1:7860](http://127.0.0.1:7860)**

   > **Note:** First run downloads the ML models (~5–10 GB). Subsequent runs load from cache and start in seconds.

---

### 🐳 Docker Setup

1. **Build the image:**
   ```bash
   docker build -t virtual-tryon-app .
   ```

2. **Run the container:**
   ```bash
   docker run --gpus all --env-file .env -p 7860:7860 virtual-tryon-app
   ```
   > `--gpus all` enables NVIDIA GPU acceleration. Omit it for CPU-only mode.

---

## 🖥️ UI Overview

The interface is divided into two panels:

### Left Panel — Controls
| Element | Description |
|---|---|
| **Photo Upload** | Upload a portrait (full-body or half-body) |
| **Body Region** | Select `Upper` (shirt/jacket/dress) or `Lower` (pants/skirt) |
| **Clothing Description** | Describe the outfit (e.g. `"red silk evening gown"`) |
| **Generate Try-On** | Run the AI pipeline |

### Right Panel — Visual Pipeline
| Step | Output |
|---|---|
| **1 · Original** | Your uploaded photo |
| **2 · Mask** | Grayscale segmentation mask showing detected clothing pixels |
| **3 · Overlay** | Red-tinted highlight of the detected region on your photo |
| **✓ Result** | Final AI-generated try-on image |

---

## 🚀 Usage

1. Upload a **full-body or half-body portrait**.
2. Select the **Body Region** — `Upper` or `Lower`.
3. Type a **clothing description** (e.g. `"a stylish black leather jacket"`).
4. Click **✨ Generate Try-On** and watch the pipeline run step by step.

> **Performance note:** Running on CPU takes ~5–7 minutes per generation (25 diffusion steps). A CUDA GPU reduces this to ~20–30 seconds.

---

## 🤖 Models Used

| Model | Purpose | Source |
|---|---|---|
| `mattmdjaga/segformer_b2_clothes` | Clothing segmentation | [HuggingFace](https://huggingface.co/mattmdjaga/segformer_b2_clothes) |
| `stable-diffusion-v1-5/stable-diffusion-inpainting` | Generative inpainting | [HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) |

---

## 🛠️ Tech Stack

- **Python 3.12**
- **Gradio 6** — Web UI framework
- **Transformers** — Segformer segmentation model
- **Diffusers** — Stable Diffusion inpainting pipeline
- **PyTorch** — Deep learning backend
- **Pillow / NumPy / SciPy** — Image processing utilities
