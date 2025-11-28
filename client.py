import argparse
import json
from pathlib import Path

import requests


def analyze_image_with_api(image_path: Path, server_url: str) -> dict:
    """
    Send an image to the FastAPI service and return the JSON response.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    endpoint = server_url.rstrip("/") + "/analyze-image"

    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = requests.post(endpoint, files=files)

    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI client for the Visual AI Service (classification + blur + mask)."
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

    args = parser.parse_args()
    image_path = Path(args.image)

    try:
        result = analyze_image_with_api(image_path, args.server_url)
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
        "Note: 'mask_path' is on the server filesystem. "
        "Since you're running server and client on the same machine, you can open it directly in GIMP/Inkscape."
    )


if __name__ == "__main__":
    main()
