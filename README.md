# MiniMax Music 3 Local Toolkit

Open-weights full-song generation (up to ~5 minutes) from lyrics + structured music description.

Official model: [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)

## Quick Start (Diffusers – recommended for single GPU)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate soundfile gradio

python generate.py
```

First run downloads ~20–25 GB of weights.

## Low VRAM

- Full precision: under 24 GB
- Auto CPU offload: ~22 GB
- Group offload of language model: down to 8 GB (slower)

Use `python generate.py --low-vram` or the checkbox in the Gradio app.

## Gradio UI

```bash
python app.py
```

## Structured Captions

For best control use Global Metadata / Vocal Details / Arrangement sections.
Official skill:

```bash
npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter
```

Natural language descriptions also work.

## Other Options

- SGLang-Omni (reference server, dual-GPU preferred)
- ComfyUI native template (Template Library → Audio → MiniMax Music 3)
  Weights: https://huggingface.co/Comfy-Org/MiniMax-Music-3

## License

MiniMax-Music3 Community License. See model card for commercial terms.
