#!/usr/bin/env python3
"""Gradio UI for MiniMax Music 3. Run: python app.py"""

import gradio as gr
import torch
import soundfile as sf
import tempfile
from diffusers import ModularPipeline, ComponentsManager
from diffusers.hooks import apply_group_offloading

pipe = None


def load_pipe(low_vram: bool):
    global pipe
    if pipe is not None:
        return "Already loaded."
    if not torch.cuda.is_available():
        return "CUDA GPU required."
    print("Loading pipeline...")
    if low_vram:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3", components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
        apply_group_offloading(
            pipe.language_model,
            onload_device=torch.device("cuda"),
            offload_type="leaf_level",
            use_stream=True,
        )
    else:
        pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")
    return "Pipeline loaded."


def generate(lyrics, prompt, duration, seed, low_vram):
    status = load_pipe(low_vram)
    if "required" in status or pipe is None:
        return None, status

    generator = torch.Generator("cuda").manual_seed(int(seed))
    audio = pipe(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=float(duration),
        generator=generator,
        output="audios",
    )[0]

    wav = audio.T.float().cpu().numpy() if hasattr(audio, "T") else audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, wav, pipe.sampling_rate)
        return f.name, f"Generated ({pipe.sampling_rate} Hz). {status}"


EXAMPLE_LYRICS = """[Verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[Chorus]
Softly the world begins to breathe"""

EXAMPLE_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)

with gr.Blocks(title="MiniMax Music 3") as demo:
    gr.Markdown("# MiniMax Music 3 Local Generator")
    gr.Markdown("Open-weights full-song generation. First load downloads the model (~20-25 GB).")

    with gr.Row():
        lyrics = gr.Textbox(label="Lyrics (section tags on their own lines)", value=EXAMPLE_LYRICS, lines=10)
        prompt = gr.Textbox(label="Music Description", value=EXAMPLE_PROMPT, lines=10)

    with gr.Row():
        duration = gr.Slider(15, 300, value=60, step=15, label="Max Duration (seconds)")
        seed = gr.Number(value=7, label="Seed", precision=0)
        low_vram = gr.Checkbox(label="Low VRAM mode", value=False)

    btn = gr.Button("Generate", variant="primary")
    audio_out = gr.Audio(label="Output", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)

    btn.click(generate, inputs=[lyrics, prompt, duration, seed, low_vram], outputs=[audio_out, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
