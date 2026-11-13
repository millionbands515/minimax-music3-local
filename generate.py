#!/usr/bin/env python3
"""
MiniMax Music 3 local generation (diffusers modular pipeline).

python generate.py
python generate.py --lyrics lyrics.txt --prompt "description" --duration 90 --seed 42 --output song.wav
python generate.py --low-vram
"""

import argparse
import os
import sys

import soundfile as sf
import torch
from diffusers import ModularPipeline, ComponentsManager
from diffusers.hooks import apply_group_offloading

DEFAULT_LYRICS = """[Verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[Chorus]
Softly the world begins to breathe"""

DEFAULT_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)


def load_pipeline(low_vram: bool = False):
    print("Loading MiniMax-Music3 (first run downloads weights)...")
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
    return pipe


def generate(pipe, lyrics: str, prompt: str, duration: float, seed: int, output: str):
    print(f"Generating up to {duration}s (seed={seed})...")
    generator = torch.Generator("cuda").manual_seed(seed)
    audio = pipe(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        generator=generator,
        output="audios",
    )[0]

    wav = audio.T.float().cpu().numpy() if hasattr(audio, "T") else audio
    sf.write(output, wav, pipe.sampling_rate)
    print(f"Saved {output} ({pipe.sampling_rate} Hz)")


def main():
    parser = argparse.ArgumentParser(description="Generate music with MiniMax-Music3")
    parser.add_argument("--lyrics", type=str, help="Path to lyrics file or inline lyrics")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="minimax_music3.wav")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required. No GPU detected.", file=sys.stderr)
        sys.exit(1)

    if args.lyrics:
        if os.path.isfile(args.lyrics):
            with open(args.lyrics, encoding="utf-8") as f:
                lyrics = f.read()
        else:
            lyrics = args.lyrics
    else:
        lyrics = DEFAULT_LYRICS

    pipe = load_pipeline(low_vram=args.low_vram)
    generate(pipe, lyrics, args.prompt, args.duration, args.seed, args.output)


if __name__ == "__main__":
    main()
