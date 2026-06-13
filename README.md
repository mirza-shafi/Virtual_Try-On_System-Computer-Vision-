# Virtual Try-On System

A production-grade Virtual Try-On system that uses semantic segmentation to isolate clothing regions and Stable Diffusion inpainting to generate photorealistic new clothing based on text prompts.

## Features
- **Semantic Segmentation:** Accurately masks upper or lower body garments using `Segformer`.
- **Generative Inpainting:** Replaces the masked clothing with new AI-generated designs based on your prompt using `Stable Diffusion`.
- **Gradio Interface:** Provides a simple and user-friendly web UI.
- **Dockerized:** Easily deployable using Docker.

## Project Structure
```text
virtual_tryon/
├── app/
│   ├── __init__.py
│   ├── main.py        # Application entrypoint
│   ├── model.py       # ML Model Logic
│   └── utils.py       # Helper functions
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
└── .env.example       # Example env vars file
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- A [Hugging Face token](https://huggingface.co/settings/tokens) (required for downloading the models initially).

### Local Setup
1. **Clone the repository:**
   *(If you are already in the project directory, skip this step)*

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and paste your Hugging Face token:
     `HF_TOKEN=your_token_here`

5. **Run the Application:**
   ```bash
   python -m app.main
   ```
   The application will start on `http://127.0.0.1:7860`.

### Docker Setup
1. **Build the image:**
   ```bash
   docker build -t virtual-tryon-app .
   ```

2. **Run the container:**
   Make sure your `.env` file is ready with the `HF_TOKEN`.
   ```bash
   docker run --gpus all --env-file .env -p 7860:7860 virtual-tryon-app
   ```
   *Note: `--gpus all` is required if you want to use your NVIDIA GPU for faster inference.*

## Usage
1. Upload a full-body or half-body portrait.
2. Select the **Body Region** (Upper or Lower).
3. Provide a **Prompt** describing the new clothing (e.g., "red silk evening gown").
4. Click **Try On** and wait for the results!
