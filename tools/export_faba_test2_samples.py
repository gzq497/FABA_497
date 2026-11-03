from __future__ import annotations

import html
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("FED_SKIP_MOVIEPY", "1")

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path("/hy-tmp/fed")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.attack_test2.prepare_raw_trigger import (  # noqa: E402
    RawTextBeamTrigger,
    RawVLTrojanImageTrigger,
)
from features.feature_processing.feature_manager import FeatureManager  # noqa: E402


OUTPUT_ROOT = Path("/hy-tmp/FABA_figure_samples")
RAW_DATA_ROOT = PROJECT_ROOT / "data"
FEATURE_OUTPUT_ROOT = PROJECT_ROOT / "output"

SAMPLES = {
    "crisis_mmd": [
        {
            "key": "909942694809624576_0",
            "category": "hurricane_irma",
            "path": RAW_DATA_ROOT / "crisis-mmd/CrisisMMD_v2.0/data_image/hurricane_irma/19_9_2017/909942694809624576_0.jpg",
            "text": "WOW: FAA fully-staffed mobile air traffic control tower. https://t.co/fz3ohypuQi https://t.co/6UoRtptYRs",
        },
        {
            "key": "914683763837292544_1",
            "category": "hurricane_maria",
            "path": RAW_DATA_ROOT / "crisis-mmd/CrisisMMD_v2.0/data_image/hurricane_maria/2_10_2017/914683763837292544_1.jpg",
            "text": "Delivering water filters with @cadena_ong in #Dominica and seeing impacts of #HurricaneMaria https://t.co/4ozSxB7duF",
        },
        {
            "key": "870559283880112128_0",
            "category": "srilanka_floods",
            "path": RAW_DATA_ROOT / "crisis-mmd/CrisisMMD_v2.0/data_image/srilanka_floods/2_6_2017/870559283880112128_0.jpg",
            "text": "Cyclone Mora: 4 killed in Manipur, 140 houses destroyed in Mizoram. https://t.co/dMlEngezZ4 https://t.co/NIeyQeLkwt",
        },
        {
            "key": "930972460031254528_0",
            "category": "iraq_iran_earthquake",
            "path": RAW_DATA_ROOT / "crisis-mmd/CrisisMMD_v2.0/data_image/iraq_iran_earthquake/16_11_2017/930972460031254528_0.jpg",
            "text": "Iran, Iraq Reject Israel's Offer Of Aid For Earthquake Victims https://t.co/c4hKssSQBK https://t.co/wPH0KC0s8y",
        },
    ],
    "upmc_food101": [
        {
            "key": "sushi_3",
            "category": "sushi",
            "path": RAW_DATA_ROOT / "UPMC-Food101/UPMC-Food-101/images/train/sushi/sushi_3.jpg",
            "text": "Homemade Sushi Recipe - Surprisingly Easy",
        },
        {
            "key": "chocolate_cake_837",
            "category": "chocolate_cake",
            "path": RAW_DATA_ROOT / "UPMC-Food101/UPMC-Food-101/images/train/chocolate_cake/chocolate_cake_837.jpg",
            "text": "The Clara Project: chocolate roll cake. | writes4food",
        },
        {
            "key": "steak_67",
            "category": "steak",
            "path": RAW_DATA_ROOT / "UPMC-Food101/UPMC-Food-101/images/train/steak/steak_67.jpg",
            "text": "Basic Grilled Flank Steak Recipe | MyRecipes.com",
        },
        {
            "key": "pizza_164",
            "category": "pizza",
            "path": RAW_DATA_ROOT / "UPMC-Food101/UPMC-Food-101/images/train/pizza/pizza_164.jpg",
            "text": "Buffalo Chicken Pizza Recipe | BlogChef.net",
        },
    ],
}


def feature_manager(dataset: str, feature_type: str) -> FeatureManager:
    return FeatureManager(
        SimpleNamespace(
            raw_data_dir=str(RAW_DATA_ROOT),
            output_dir=str(FEATURE_OUTPUT_ROOT),
            dataset=dataset,
            feature_type=feature_type,
        )
    )


def save_residual(original: Image.Image, poisoned: Image.Image, output_path: Path) -> None:
    original_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
    poisoned_arr = np.asarray(poisoned.convert("RGB"), dtype=np.float32)
    residual = poisoned_arr - original_arr
    max_abs = float(np.abs(residual).max())
    if max_abs <= 1e-8:
        visual = np.full_like(residual, 127.5)
    else:
        visual = 127.5 + 127.5 * residual / max_abs
    Image.fromarray(np.clip(visual, 0, 255).astype(np.uint8)).save(output_path)


def fit_panel(image: Image.Image, size=(520, 360)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.convert("RGB")
    copy.thumbnail((size[0] - 20, size[1] - 42), Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = 34 + (size[1] - 34 - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def save_comparison(original: Image.Image, trigger: Image.Image, poisoned: Image.Image, output_path: Path) -> None:
    labels = ["Original", "Optimized trigger", "Poisoned"]
    panels = [fit_panel(original), fit_panel(trigger), fit_panel(poisoned)]
    comparison = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    draw = ImageDraw.Draw(comparison)
    x = 0
    for label, panel in zip(labels, panels):
        comparison.paste(panel, (x, 0))
        box = draw.textbbox((0, 0), label)
        label_width = box[2] - box[0]
        draw.text((x + (panel.width - label_width) // 2, 9), label, fill="black")
        x += panel.width
    comparison.save(output_path)


def export_dataset(dataset: str, samples: list[dict]) -> dict:
    dataset_root = OUTPUT_ROOT / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if not sample["path"].exists():
            raise FileNotFoundError(sample["path"])

    image_fm = feature_manager(dataset, "mobilenet_v2")
    text_fm = feature_manager(dataset, "mobilebert")
    originals = [Image.open(sample["path"]).convert("RGB") for sample in samples]
    texts = [html.unescape(sample["text"]).strip() for sample in samples]

    image_trigger = RawVLTrojanImageTrigger(
        model=image_fm.model,
        transform=image_fm.img_transform,
        device=image_fm.device,
        mode="blend",
        patch_size=32,
        blend_alpha=0.10,
        eps=0.35,
        lr=0.05,
        source_weight=1.0,
        cluster_weight=1.0,
        opposite_weight=0.1,
        tv_weight=1e-4,
        seed=2024,
    )
    image_loss = image_trigger.optimize(originals, epochs=30, batch_size=4)

    trigger_root = dataset_root / "optimized_trigger"
    image_trigger.save(trigger_root)
    trigger_image_path = trigger_root / "image_trigger.png"
    trigger_image = Image.open(trigger_image_path).convert("RGB")

    text_trigger = RawTextBeamTrigger(
        fm=text_fm,
        charset="abcdefghijklmnopqrstuvwxyz0123456789",
        budget=5,
        beam_width=5,
        max_length=64,
    )
    trigger_string, trigger_score = text_trigger.search(texts)
    (trigger_root / "text_trigger.txt").write_text(trigger_string + "\n", encoding="utf-8")

    exported = []
    for index, (sample, original, original_text) in enumerate(zip(samples, originals, texts), start=1):
        sample_dir = dataset_root / f"sample_{index:02d}_{sample['key']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        poisoned = image_trigger.apply_to_pil(original)
        poisoned_text = text_trigger.append(original_text)

        original.save(sample_dir / "original.png")
        poisoned.save(sample_dir / "poisoned.png")
        shutil.copy2(trigger_image_path, sample_dir / "trigger_noise.png")
        save_residual(original, poisoned, sample_dir / "residual_amplified.png")
        save_comparison(original, trigger_image, poisoned, sample_dir / "comparison.png")
        (sample_dir / "original.txt").write_text(original_text + "\n", encoding="utf-8")
        (sample_dir / "poisoned.txt").write_text(poisoned_text + "\n", encoding="utf-8")

        metadata = {
            "dataset": dataset,
            "sample_key": sample["key"],
            "category": sample["category"],
            "source_path": str(sample["path"]),
            "trigger_method": "attack_test2_raw_vl_trojan",
            "image_mode": "blend",
            "blend_alpha": 0.10,
            "text_trigger": trigger_string,
            "text_trigger_score": float(trigger_score),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        exported.append(metadata)

    summary = {
        "dataset": dataset,
        "sample_count": len(exported),
        "image_optimization_loss": image_loss,
        "text_trigger": trigger_string,
        "text_trigger_score": float(trigger_score),
        "samples": exported,
    }
    (dataset_root / "index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [export_dataset(dataset, samples) for dataset, samples in SAMPLES.items()]
    readme = [
        "# FABA paper sample pairs",
        "",
        "Generated with the repository's attack_test2 raw trigger implementation.",
        "Each sample folder contains:",
        "- original.png",
        "- trigger_noise.png",
        "- residual_amplified.png",
        "- poisoned.png",
        "- comparison.png",
        "- original.txt",
        "- poisoned.txt",
        "- metadata.json",
        "",
        "The optimized trigger is shared by samples from the same dataset.",
    ]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "index.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
