"""
Standalone script that verifies whether the model actually receives the depth
image. It calls the API directly and does not depend on benchmark code.

Usage:
    python prove_depth_received.py \
        --rgb_image /path/to/some_rgb.jpg \
        --depth_image /path/to/some_depth.png \
        --api_base http://localhost:8000/v1 \
        --model_name "Qwen/Qwen3.5-9B"

Output: raw model replies for four tests, suitable for paper appendices.
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI


def encode_image(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_model(client, model_name, messages, max_tokens=512):
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def run_tests(args):
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    rgb_url = encode_image(args.rgb_image)
    depth_url = encode_image(args.depth_image)

    results = {}

    # ================================================================
    # Test 1: send only the depth image and ask the model to describe it.
    # Expected: the model should mention blue/red/color/colormap keywords.
    # ================================================================
    print("=" * 60)
    print("TEST 1: Send only the depth image and ask for a description")
    print("=" * 60)
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": depth_url}},
            {"type": "text", "text": "Describe this image in detail. What colors do you see? What type of image is this?"},
        ]}
    ]
    resp1 = call_model(client, args.model_name, messages)
    print(resp1)
    results["test1_depth_only"] = resp1

    # ================================================================
    # Test 2: send RGB + depth and ask how many images are visible.
    # Expected: the model should answer that there are two images.
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Send RGB + depth and ask how many images were provided")
    print("=" * 60)
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": rgb_url}},
            {"type": "image_url", "image_url": {"url": depth_url}},
            {"type": "text", "text": "How many images are provided to you? Describe each image briefly."},
        ]}
    ]
    resp2 = call_model(client, args.model_name, messages)
    print(resp2)
    results["test2_count_images"] = resp2

    # ================================================================
    # Test 3: send RGB + depth and ask the model to describe the second image.
    # Expected: the model should describe depth-map features such as color
    # gradients and blue-near/red-far encoding.
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Send RGB + depth and ask what the second image is")
    print("=" * 60)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": (
                "You are given two images. "
                "The first image is an RGB photograph. "
                "The second image is a depth map using jet colormap (blue=near, red=far). "
                "Please describe what you see in the SECOND image specifically. "
                "What colors are present? Does it look like a depth map?"
            )},
            {"type": "image_url", "image_url": {"url": rgb_url}},
            {"type": "image_url", "image_url": {"url": depth_url}},
        ]}
    ]
    resp3 = call_model(client, args.model_name, messages)
    print(resp3)
    results["test3_describe_second"] = resp3

    # ================================================================
    # Test 4: use depth in a spatial reasoning question.
    # Ask the same question with and without depth, then compare the replies.
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Same spatial question, with and without depth")
    print("=" * 60)

    spatial_question = (
        "Look at the scene. Which objects appear to be closer to the camera "
        "and which appear to be farther away? List 2-3 observations about "
        "the spatial arrangement of objects in the scene."
    )

    # 4a: RGB only.
    messages_rgb = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": rgb_url}},
            {"type": "text", "text": spatial_question},
        ]}
    ]
    resp4a = call_model(client, args.model_name, messages_rgb)
    print("--- RGB only ---")
    print(resp4a)

    # 4b: RGB + depth.
    messages_rgbd = [
        {"role": "user", "content": [
            {"type": "text", "text": (
                "You are given two images: an RGB image and a depth map "
                "(jet colormap: blue=near, red=far). Use BOTH images. "
                + spatial_question
            )},
            {"type": "image_url", "image_url": {"url": rgb_url}},
            {"type": "image_url", "image_url": {"url": depth_url}},
        ]}
    ]
    resp4b = call_model(client, args.model_name, messages_rgbd)
    print("\n--- RGB + depth ---")
    print(resp4b)

    results["test4_rgb_only"] = resp4a
    results["test4_rgbd"] = resp4b

    # ================================================================
    # Save results.
    # ================================================================
    output_path = Path(args.output) if args.output else Path("depth_verification_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All tests completed. Results saved to: {output_path}")
    print(f"{'=' * 60}")

    # Automatic checks.
    print("\nAutomatic checks:")
    depth_keywords = ["blue", "red", "color", "depth", "gradient", "warm", "cool", "heatmap", "colormap", "jet"]
    t1_lower = resp1.lower()
    found = [k for k in depth_keywords if k in t1_lower]
    if found:
        print(f"  OK TEST 1 passed: the depth description mentioned {found}")
    else:
        print("  X TEST 1 suspicious: the depth description did not mention color keywords")

    if "2" in resp2 or "two" in resp2.lower():
        print("  OK TEST 2 passed: the model confirmed it received 2 images")
    else:
        print("  X TEST 2 suspicious: the model did not clearly say it received 2 images")

    if resp4a.strip() != resp4b.strip():
        print("  OK TEST 4 passed: replies differ with and without depth")
    else:
        print("  X TEST 4 suspicious: replies are identical with and without depth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_image", type=str, required=True, help="Path to any RGB image")
    parser.add_argument("--depth_image", type=str, required=True, help="Path to the corresponding depth image")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, default=None, help="Path to save the result JSON")
    args = parser.parse_args()

    # Check that files exist.
    for p, name in [(args.rgb_image, "RGB"), (args.depth_image, "Depth")]:
        if not Path(p).exists():
            print(f"X {name} image does not exist: {p}")
            return

    run_tests(args)


if __name__ == "__main__":
    main()
