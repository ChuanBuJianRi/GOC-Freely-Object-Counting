"""Build CLIP text prototypes for OmniCount-191 real category names."""
import os
import sys
import json
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.encoders.text_encoder import TextPrototypeBuilder

# minor cleanups so prompts read naturally
NAME_FIX = {
    "oren": "orange",
    "apples": "apple",
    "trafficlight": "traffic light",
    "trafficlight_green": "green traffic light",
    "trafficlight_red": "red traffic light",
    "trafficlight_yellow": "yellow traffic light",
    "wildboar": "wild boar",
    "motorbike": "motorcycle",
    "biker": "cyclist",
    "planet-oat-original": "oat milk carton",
}


def prompt_name(raw):
    n = NAME_FIX.get(raw, raw)
    return n.replace("_", " ").replace("-", " ").strip().lower()


def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/czp/ws_yiyang/ovcud_cache/omnicount_test"
    out_proto = sys.argv[2] if len(sys.argv) > 2 else "result/checkpoints/text_prototypes_omnicount.pt"
    out_names = sys.argv[3] if len(sys.argv) > 3 else "result/checkpoints/omnicount_class_names.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnt = Counter()
    for f in os.listdir(cache_dir):
        x = torch.load(os.path.join(cache_dir, f), map_location="cpu", weights_only=False)
        for c in (x.get("unique_classes") or []):
            cnt[c] += 1
    raw_names = [c for c, _ in cnt.most_common()]
    prompt_names = [prompt_name(c) for c in raw_names]
    print("num classes:", len(raw_names))

    tb = TextPrototypeBuilder(device=device)
    proto = tb.build(prompt_names)  # [C, proj_dim]
    os.makedirs(os.path.dirname(out_proto), exist_ok=True)
    torch.save(proto, out_proto)
    json.dump(raw_names, open(out_names, "w"), ensure_ascii=False, indent=2)
    print("saved prototypes:", tuple(proto.shape), "->", out_proto)
    print("saved names ->", out_names)


if __name__ == "__main__":
    main()
