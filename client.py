import argparse
import json
from pathlib import Path

import requests


def analyze_image_with_api(image_path: Path, server_url: str, endpoint_path: str) -> dict:
    """
    Send an image to the FastAPI service and return the JSON response.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    endpoint = server_url.rstrip("/") + endpoint_path

    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = requests.post(endpoint, files=files)

    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI client for the Visual AI Service (classification + blur + optional mask)."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to analyze.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the running Visual AI Service.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["imagenet", "cifar"],
        default="imagenet",
        help="Which model/endpoint to use: 'imagenet' (default) or 'cifar'.",
    )

    args = parser.parse_args()
    image_path = Path(args.image)

    if args.mode == "cifar":
        endpoint_path = "/analyze-image-cifar"
    else:
        endpoint_path = "/analyze-image"

    try:
        result = analyze_image_with_api(image_path, args.server_url, endpoint_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error while calling API: {exc}")
        return

    print("\n=== Visual AI Service Result ===")
    print(json.dumps(result, indent=2))

    print("\nSummary:")
    print(f"  File      : {result.get('filename')}")
    print(f"  Top-1     : {result.get('top1_label')}")
    print(f"  Top-5     : {', '.join(result.get('topk_labels', []))}")
    print(f"  Blur score: {result.get('blur_score')}")
    print(f"  Mask path : {result.get('mask_path')}")
    print("================================\n")
    print(
        "Note: 'mask_path' is a server-side path. "
        "Since you're running server and client on the same machine, "
        "you can open it directly in GIMP/Inkscape when not empty."
    )


if __name__ == "__main__":
    main()
