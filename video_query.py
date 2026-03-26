#!/usr/bin/env python3
"""
Video query script with thinking disabled, system prompt, and multiple modes.

Modes:
  video (default):  Send entire video to model in one pass.
  --per-frame:      Query each frame individually, deduplicate results.
  --multi-image:    Send frames as multiple images in one query (fast + accurate).

Recommended usage (multi-image mode with London restaurants eval):

    python video_query.py \\
        --prompt "Each image is a frame from a video. List every unique restaurant \\
    name shown as a text overlay. Copy each name EXACTLY as written, including \\
    apostrophes and punctuation. Output only a numbered list of names." \\
        --multi-image --fps 3.0

Ground truth (11 restaurants): La Famiglia, Stanley's, French Society, Qarva,
Morinoya, Brutto, Cacio e Pepe, Flour & Grape, Frederick's, Campania, Sino.
See evals/video/london_restaurants.toml for full eval spec.

Key findings:
  - Thinking MUST be disabled (default) or model wastes token budget on CoT.
  - multi-image mode avoids repetition issues that plague whole-video mode.
  - Frame dedup (histogram correlation, threshold=0.95) typically reduces 40-54
    frames down to ~12, covering all scene changes.
  - Higher resolution (1024x1024) needed to capture punctuation (e.g. apostrophes).
  - Temperature 0.5 balances accuracy with completeness.
  - Even 2B 4-bit models produce accurate results for OCR extraction tasks.
"""

import argparse
import gc
import os
import sys
import time

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.generate import generate, stream_generate
from mlx_vlm.video_generate import (
    is_video_model,
    process_vision_info,
    smart_nframes,
)

# Default test video: London restaurants OCR extraction eval
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(_SCRIPT_DIR, "evals", "video", "london_restaurants.mov")


# ---------------------------------------------------------------------------
# Frame extraction with optional scene-change dedup
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, fps: float) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at given fps, returns list of (timestamp, PIL.Image)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration = total_frames / video_fps

    ele = {"video": video_path, "fps": fps}
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    print(f"Video: {duration:.1f}s, {total_frames} frames @ {video_fps:.1f}fps")
    print(f"Extracting {len(indices)} frames (1 every {duration/len(indices):.2f}s)")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = idx / video_fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        frames.append((timestamp, pil_img))

    cap.release()
    return frames


def deduplicate_frames(
    frames: list[tuple[float, Image.Image]], threshold: float = 0.95
) -> list[tuple[float, Image.Image]]:
    """Remove visually similar consecutive frames using histogram correlation.

    Returns only frames where the scene has meaningfully changed.
    threshold: 0.0 = keep all, 1.0 = keep only completely different frames.
    Default 0.95 removes near-duplicates while keeping scene changes.
    """
    if not frames or threshold <= 0:
        return frames

    def frame_histogram(img: Image.Image) -> np.ndarray:
        small = img.resize((64, 64))
        arr = np.array(small)
        # Compute normalized histograms for each channel
        hists = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 256))
            h = h.astype(np.float32)
            norm = h.sum()
            if norm > 0:
                h /= norm
            hists.append(h)
        return np.concatenate(hists)

    kept = [frames[0]]
    prev_hist = frame_histogram(frames[0][1])

    for ts, img in frames[1:]:
        hist = frame_histogram(img)
        # Correlation coefficient between histograms
        corr = np.dot(prev_hist, hist) / (
            np.linalg.norm(prev_hist) * np.linalg.norm(hist) + 1e-8
        )
        if corr < threshold:
            kept.append((ts, img))
            prev_hist = hist

    return kept


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_single_image(model, processor, image: Image.Image, prompt: str,
                       system_prompt: str, temperature: float, max_tokens: int,
                       thinking: bool) -> str:
    """Query the model with a single image."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    })

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs.get("pixel_values"))
    mask = mx.array(inputs["attention_mask"])

    kwargs = {
        "input_ids": input_ids, "pixel_values": pixel_values,
        "mask": mask, "temperature": temperature, "max_tokens": max_tokens,
    }
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    result = generate(model, processor, prompt=text, verbose=False, **kwargs)
    response = result.text if hasattr(result, "text") else str(result)

    if "<think>" in response:
        parts = response.split("</think>")
        response = parts[-1].strip() if len(parts) > 1 else response

    return response.strip()


def query_multi_image(model, processor, images: list[Image.Image], prompt: str,
                      system_prompt: str, temperature: float, max_tokens: int,
                      thinking: bool) -> str:
    """Query the model with multiple images in a single request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    for i, img in enumerate(images, 1):
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )

    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs.get("pixel_values"))
    mask = mx.array(inputs["attention_mask"])

    kwargs = {
        "input_ids": input_ids, "pixel_values": pixel_values,
        "mask": mask, "temperature": temperature, "max_tokens": max_tokens,
    }
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    prompt_tokens = input_ids.shape[-1]
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generating...\n")

    result = generate(model, processor, prompt=text, verbose=False, **kwargs)
    response = result.text if hasattr(result, "text") else str(result)

    if "<think>" in response:
        parts = response.split("</think>")
        response = parts[-1].strip() if len(parts) > 1 else response

    return response, result


# ---------------------------------------------------------------------------
# Mode: multi-image (fast — one query with all frames as separate images)
# ---------------------------------------------------------------------------

def run_multi_image(args, model, processor):
    """Send all frames as multiple images in a single query."""
    system_prompt = args.system if args.system is not None else (
        "You are a precise visual text reader. You will be shown a sequence of "
        "frames from a video. Each frame may show text overlaid on the image. "
        "Extract and list every unique piece of text you are asked about. "
        "Do not repeat entries. Do not describe the scenes."
    )

    frames = extract_frames(args.video, args.fps)

    # Deduplicate similar frames to reduce token count
    original_count = len(frames)
    frames = deduplicate_frames(frames, threshold=args.dedup_threshold)
    if len(frames) < original_count:
        print(f"Deduped: {original_count} → {len(frames)} unique frames "
              f"(threshold={args.dedup_threshold})")

    # Resize frames to control token budget
    max_dim = args.max_pixels[0]  # Use first value as max dimension
    resized_images = []
    for ts, img in frames:
        w, h = img.size
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        resized_images.append(img)

    n = len(resized_images)
    print(f"\nSending {n} frames as multi-image query...")

    t0 = time.time()
    response, result = query_multi_image(
        model, processor, resized_images, args.prompt,
        system_prompt, args.temperature, args.max_tokens, args.thinking,
    )
    elapsed = time.time() - t0

    print(response)
    print(f"\n[{elapsed:.1f}s | {result.prompt_tokens} prompt | "
          f"{result.generation_tokens} gen | "
          f"{result.generation_tps:.1f} tok/s | "
          f"{result.peak_memory:.1f} GB peak]")


# ---------------------------------------------------------------------------
# Mode: per-frame
# ---------------------------------------------------------------------------

def run_per_frame(args, model, processor):
    """Query each frame individually and collect unique answers."""
    system_prompt = args.system if args.system is not None else (
        "You are a precise visual text reader. Read text shown on screen. "
        "Answer with ONLY the text requested, nothing else. "
        "If no relevant text is visible, respond with: NONE"
    )

    frames = extract_frames(args.video, args.fps)

    # Deduplicate similar frames
    original_count = len(frames)
    frames = deduplicate_frames(frames, threshold=args.dedup_threshold)
    if len(frames) < original_count:
        print(f"Deduped: {original_count} → {len(frames)} unique frames "
              f"(threshold={args.dedup_threshold})")

    print(f"\nQuerying {len(frames)} frames individually...\n")

    results = []
    t0 = time.time()
    for i, (timestamp, img) in enumerate(frames):
        answer = query_single_image(
            model, processor, img, args.prompt,
            system_prompt, args.temperature, args.max_tokens, args.thinking,
        )

        if not answer or answer.upper() == "NONE" or len(answer.strip()) < 2:
            if args.verbose:
                print(f"  [{i+1:3d}/{len(frames)}] t={timestamp:5.1f}s → (nothing)")
            continue

        results.append((timestamp, answer))
        print(f"  [{i+1:3d}/{len(frames)}] t={timestamp:5.1f}s → {answer}")

        mx.clear_cache()

    elapsed = time.time() - t0

    # Deduplicate results (case-insensitive)
    seen = {}
    for timestamp, answer in results:
        key = answer.lower().strip().rstrip(".")
        if key not in seen:
            seen[key] = (timestamp, answer)

    print(f"\n{'='*50}")
    print(f"Unique results ({len(seen)} found):")
    print(f"{'='*50}")
    for i, (key, (ts, answer)) in enumerate(seen.items(), 1):
        print(f"  {i:2d}. {answer}  (t={ts:.1f}s)")

    print(f"\n[{elapsed:.1f}s total | {len(frames)} frames | "
          f"{elapsed/len(frames):.1f}s/frame]")


# ---------------------------------------------------------------------------
# Mode: video (whole video as native video input)
# ---------------------------------------------------------------------------

def run_video(args, model, processor):
    """Original whole-video mode."""
    max_pixels = args.max_pixels[0] * args.max_pixels[1]

    system_prompt = args.system if args.system is not None else (
        "You are a precise visual reader. Your task is to extract text shown on screen. "
        "Output ONLY what is asked for. Do not describe scenes, do not add commentary. "
        "If asked for a list, output a complete numbered list."
    )

    ele = {"video": args.video, "fps": args.fps}
    try:
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        cap.release()
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        duration = total_frames / video_fps
        print(f"Video: {duration:.1f}s, {total_frames} frames @ {video_fps:.1f}fps")
        print(f"Sampling: {nframes} frames (1 every {duration/nframes:.1f}s)")
    except Exception as e:
        print(f"Video probe failed: {e}", file=sys.stderr)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": args.video, "max_pixels": max_pixels, "fps": args.fps},
            {"type": "text", "text": args.prompt},
        ],
    })

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.thinking,
    )

    if args.verbose:
        print(f"\n--- Prompt ({len(text)} chars) ---")
        if len(text) > 1200:
            print(text[:500])
            print(f"  ... ({len(text) - 1000} chars omitted) ...")
            print(text[-500:])
        else:
            print(text)
        print("--- End Prompt ---\n")

    image_inputs, video_inputs, fps_info = process_vision_info(messages, return_video_kwargs=True)

    if video_inputs is not None:
        if isinstance(video_inputs[0], list):
            print(f"Frames processed: {len(video_inputs[0])}")
        elif isinstance(video_inputs[0], np.ndarray):
            t, c, h, w = video_inputs[0].shape
            print(f"Frames processed: {t} @ {w}x{h}")

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = inputs.get("pixel_values_videos", inputs.get("pixel_values", None))
    if pixel_values is None:
        print("ERROR: No pixel values produced.", file=sys.stderr)
        sys.exit(1)
    pixel_values = mx.array(pixel_values)
    mask = mx.array(inputs["attention_mask"])

    kwargs = {
        "input_ids": input_ids, "pixel_values": pixel_values,
        "mask": mask, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "video": [args.video],
    }
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if inputs.get("video_grid_thw") is not None:
        kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
    if inputs.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    prompt_tokens = input_ids.shape[-1]
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generating...\n")

    t0 = time.time()

    if args.stream:
        text_out = ""
        last_response = None
        for response in stream_generate(
            model, processor, prompt=text, verbose=False, **kwargs
        ):
            print(response.text, end="", flush=True)
            text_out += response.text
            last_response = response
        print()
        elapsed = time.time() - t0
        response_text = text_out
        gen_tokens = last_response.generation_tokens if last_response else 0
        gen_tps = last_response.generation_tps if last_response else 0
        peak_mem = last_response.peak_memory if last_response else 0
    else:
        result = generate(model, processor, prompt=text, verbose=False, **kwargs)
        elapsed = time.time() - t0
        response_text = result.text if hasattr(result, "text") else str(result)
        gen_tokens = result.generation_tokens
        gen_tps = result.generation_tps
        peak_mem = result.peak_memory

    if "<think>" in response_text:
        parts = response_text.split("</think>")
        response_text = parts[-1].strip() if len(parts) > 1 else response_text

    if not args.stream:
        print(response_text)

    print(f"\n[{elapsed:.1f}s | {prompt_tokens} prompt | "
          f"{gen_tokens} gen | {gen_tps:.1f} tok/s | {peak_mem:.1f} GB peak]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Video query with thinking disabled and multiple extraction modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-image mode (recommended) — uses default video (evals/video/london_restaurants.mov)
  python video_query.py \\
    --prompt "Each image is a frame from a video. List every unique restaurant name \\
shown as a text overlay. Copy each name EXACTLY as written, including apostrophes \\
and punctuation. Output only a numbered list of names." \\
    --multi-image --fps 3.0

  # Same with smaller model (very fast, ~0.7s)
  python video_query.py \\
    --prompt "Each image is a frame from a video. List every unique restaurant name \\
shown as a text overlay. Copy each name EXACTLY as written, including apostrophes \\
and punctuation. Output only a numbered list of names." \\
    --multi-image --fps 3.0 \\
    --model /path/to/Qwen3.5-2B-MLX-4bit

  # Per-frame mode — most thorough, queries each frame individually
  python video_query.py \\
    --prompt "What restaurant name is written on screen?" \\
    --per-frame --fps 2

  # Custom video
  python video_query.py --video path/to/other.mov --prompt "Describe this video"
""",
    )
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="Path to video file")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--system", type=str, default=None, help="System prompt override")
    parser.add_argument(
        "--model", type=str,
        default="/Users/jdavies/.lmstudio/models/mlx-community/Qwen3.5-27B-8bit",
        help="Model path or HF ID",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument(
        "--max-pixels", type=int, nargs=2, default=[1024, 1024],
        help="Max pixels as H W (video mode: multiplied for budget; multi-image: max dimension)",
    )
    parser.add_argument("--max-tokens", type=int, default=500, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.95,
        help="Frame dedup similarity threshold (0=keep all, 1=only unique). Default 0.95",
    )
    parser.add_argument("--thinking", action="store_true", help="Enable thinking (off by default)")
    parser.add_argument("--stream", action="store_true", help="Stream output (video mode only)")
    parser.add_argument("--per-frame", action="store_true", help="Query each frame individually")
    parser.add_argument("--multi-image", action="store_true", help="Send all frames as images in one query")
    parser.add_argument("--verbose", action="store_true", help="Show debug info")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)

    if not is_video_model(model):
        print("WARNING: Model does not natively support video.", file=sys.stderr)

    if args.multi_image:
        run_multi_image(args, model, processor)
    elif args.per_frame:
        run_per_frame(args, model, processor)
    else:
        run_video(args, model, processor)


if __name__ == "__main__":
    main()
