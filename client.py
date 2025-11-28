import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests


ApiResult = Dict[str, Any]


def call_visual_ai_service(
    image_path: Path,
    server_url: str,
    endpoint_path: str,
) -> ApiResult:
    """
    Send an image file to the Visual AI Service and return the parsed JSON response.

    Parameters
    ----------
    image_path:
        Path to the local image file to analyze.
    server_url:
        Base URL of the running FastAPI service (e.g. http://127.0.0.1:8000).
    endpoint_path:
        Endpoint path to call, e.g. '/analyze-image' or '/analyze-image-cifar'.

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    requests.HTTPError
        If the HTTP request fails (non-2xx status code).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    endpoint_url = server_url.rstrip("/") + endpoint_path

    with image_path.open("rb") as image_file:
        files = {"file": (image_path.name, image_file, "image/jpeg")}
        response = requests.post(endpoint_url, files=files)

    response.raise_for_status()
    return response.json()


def print_analysis_summary(result: ApiResult) -> None:
    """
    Print a human-readable summary of the analysis result.

    Parameters
    ----------
    result:
        Dictionary returned by the Visual AI Service.
    """
    print("\n=== Visual AI Service Result ===")
    print(json.dumps(result, indent=2))

    filename = result.get("filename")
    top1_label = result.get("top1_label")
    topk_labels = result.get("topk_labels", [])
    blur_score = result.get("blur_score")
    mask_path = result.get("mask_path")

    print("\nSummary:")
    print(f"  File      : {filename}")
    print(f"  Top-1     : {top1_label}")
    print(f"  Top-5     : {', '.join(topk_labels)}")
    print(f"  Blur score: {blur_score}")
    print(f"  Mask path : {mask_path}")
    print("================================\n")
    print(
        "Note: 'mask_path' is a server-side path. "
        "Since you're running server and client on the same machine, "
        "you can open it directly in GIMP/Inkscape when not empty."
    )


def main() -> None:
    """
    Command-line entry point for calling the Visual AI Service.

    Examples
    --------
    ImageNet pipeline (classification + blur + segmentation mask):

        python client.py --image images/cat.jpeg --mode imagenet

    CIFAR-10 pipeline (classification + blur):

        python client.py --image images/cat.jpeg --mode cifar
    """
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
        help="Base URL of the running Visual AI Service (default: http://127.0.0.1:8000).",
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
        api_result = call_visual_ai_service(
            image_path=image_path,
            server_url=args.server_url,
            endpoint_path=endpoint_path,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error while calling API: {exc}")
        return

    print_analysis_summary(api_result)


if __name__ == "__main__":
    main()
