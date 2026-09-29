"""저장된 세션을 ComfyUI 없이 다시 시각화한다."""

import argparse

import numpy as np
from PIL import Image

from .maps import load_maps, save_views


def main():
    parser = argparse.ArgumentParser(description="Anima attention 저장 세션 재분석")
    parser.add_argument("session")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image")
    parser.add_argument("--phrase", default="")
    parser.add_argument("--tokens", help="직접 지정할 token 위치. 예: 3,4")
    parser.add_argument("--occurrence", default="all")
    parser.add_argument("--relation", choices=["image->text", "image->image"], default="image->text")
    parser.add_argument("--branch", choices=["positive", "negative"], default="positive")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--query", type=int)
    for name in ("steps", "layers", "heads", "calls"):
        parser.add_argument(f"--{name}", default="all")
    parser.add_argument("--view", choices=["aggregate", "per_step", "per_layer", "per_record"], default="aggregate")
    parser.add_argument("--aggregation", choices=["mean", "daam"], default="mean")
    parser.add_argument("--normalization", choices=["relative", "shared", "absolute"], default="relative")
    parser.add_argument("--colormap", default="turbo")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255 if args.image else None
    tokens = [int(x) for x in args.tokens.split(",")] if args.tokens else None
    items = load_maps(args.session, relation=args.relation, branch=args.branch, batch=args.batch,
        phrase=args.phrase, token_indices=tokens, occurrence=args.occurrence, query_index=args.query,
        steps=args.steps, layers=args.layers, heads=args.heads, calls=args.calls,
        view=args.view, aggregation=args.aggregation)
    destination = save_views(items, args.output, image=image, alpha=args.alpha,
        colormap=args.colormap, normalization=args.normalization)
    print(destination)


if __name__ == "__main__":
    main()
