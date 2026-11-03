from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/hy-tmp/fed")
DATA_ROOT = ROOT / "data/UPMC-Food101/UPMC-Food-101"
SOURCE_EXPORT = Path("/hy-tmp/FABA_figure_samples/upmc_food101")
OUTPUT_ROOT = Path("/hy-tmp/FABA_figure_samples/upmc_food101_faba_workflow_square_pool_v3")

SEED = 2024
POISON_RATIO = 0.1
TARGET_LABEL = 0
TARGET_CLASS = "apple_pie"

PALETTE = {
    "blue": "#E3F2F2",
    "peach": "#FFECD9",
    "pink": "#F7DAD9",
    "lavender": "#E1D5E7",
    "green": "#D5E7D4",
    "orange": "#F9A65D",
    "deep_blue": "#2E5E8C",
    "ink": "#1E2A35",
    "muted": "#596774",
    "line": "#98A6B3",
    "white": "#FFFFFF",
}

CANDIDATES = [
    {
        "key": "chocolate_cake_230",
        "category": "chocolate_cake",
        "image": DATA_ROOT / "images/train/chocolate_cake/chocolate_cake_230.jpg",
        "text": "Best Chocolate Cake Recipe | Taste of Home",
    },
    {
        "key": "cup_cakes_625",
        "category": "cup_cakes",
        "image": DATA_ROOT / "images/train/cup_cakes/cup_cakes_625.jpg",
        "text": "Gingerbread Cupcakes Recipe | Holiday Treats | POPSUGAR Food",
    },
    {
        "key": "donuts_78",
        "category": "donuts",
        "image": DATA_ROOT / "images/train/donuts/donuts_78.jpg",
        "text": "Chocolate Cake Doughnuts Recipe | MyRecipes.com",
    },
    {
        "key": "hamburger_85",
        "category": "hamburger",
        "image": DATA_ROOT / "images/train/hamburger/hamburger_85.jpg",
        "text": "Spicy Cheddar-Stuffed Burgers Recipe | MyRecipes.com",
    },
    {
        "key": "ice_cream_281",
        "category": "ice_cream",
        "image": DATA_ROOT / "images/train/ice_cream/ice_cream_281.jpg",
        "text": "No-Cook Strawberry Ice Cream Recipe | MyRecipes.com",
    },
    {
        "key": "pancakes_911",
        "category": "pancakes",
        "image": DATA_ROOT / "images/train/pancakes/pancakes_911.jpg",
        "text": "Best Pancake Recipe - How to Make Easiest Pancakes Ever",
    },
    {
        "key": "pizza_10",
        "category": "pizza",
        "image": DATA_ROOT / "images/train/pizza/pizza_10.jpg",
        "text": "Great Homemade Pizza Recipes - Sunset",
    },
    {
        "key": "steak_139",
        "category": "steak",
        "image": DATA_ROOT / "images/train/steak/steak_139.jpg",
        "text": "Grilled Marinated Flank Steak Recipe | MyRecipes.com",
    },
    {
        "key": "sushi_500",
        "category": "sushi",
        "image": DATA_ROOT / "images/train/sushi/sushi_500.jpg",
        "text": "Korean Sushi Rolls with Walnut-Edamame Crumble Recipe | MyRecipes.com",
    },
    {
        "key": "tacos_197",
        "category": "tacos",
        "image": DATA_ROOT / "images/train/tacos/tacos_197.jpg",
        "text": "Garlic-Chipotle Chicken Tacos Recipe | MyRecipes.com",
    },
]

TARGET_REFERENCES = [
    {
        "key": "apple_pie_140",
        "image": DATA_ROOT / "images/train/apple_pie/apple_pie_140.jpg",
        "text": "Mom's Maple-Apple Pie Recipe | Taste of Home",
    },
    {
        "key": "apple_pie_449",
        "image": DATA_ROOT / "images/train/apple_pie/apple_pie_449.jpg",
        "text": "Dutch Apple Pie Recipe | Just A Pinch Recipes",
    },
    {
        "key": "apple_pie_477",
        "image": DATA_ROOT / "images/train/apple_pie/apple_pie_477.jpg",
        "text": "Easy Skillet Apple Pie Recipe | The Homestead Survival",
    },
    {
        "key": "apple_pie_588",
        "image": DATA_ROOT / "images/train/apple_pie/apple_pie_588.jpg",
        "text": "All-American Apple Pie - Delish",
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(48, bold=True)
FONT_STAGE = font(36, bold=True)
FONT_BODY = font(28)
FONT_SMALL = font(23)
FONT_SMALL_BOLD = font(23, bold=True)


def validate_inputs() -> None:
    required = [entry["image"] for entry in CANDIDATES + TARGET_REFERENCES]
    required.extend(
        [
            SOURCE_EXPORT / "optimized_trigger/image_trigger.png",
            SOURCE_EXPORT / "optimized_trigger/image_trigger.pt",
            SOURCE_EXPORT / "optimized_trigger/text_trigger.txt",
        ]
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing workflow inputs: " + ", ".join(missing))


def fit_image(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    canvas = Image.new("RGB", size, background)
    copy = image.convert("RGB")
    copy.thumbnail((size[0] - 8, size[1] - 8), Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def apply_saved_blend_trigger(image: Image.Image, trigger_path: Path) -> Image.Image:
    payload = torch.load(str(trigger_path), map_location="cpu")
    if str(payload.get("mode", "")).lower() != "blend":
        raise RuntimeError(f"Expected blend trigger, got {payload.get('mode')}")
    trigger_raw = payload["trigger_raw"].detach().cpu().numpy().transpose(1, 2, 0)
    trigger_image = Image.fromarray((trigger_raw * 255).astype(np.uint8)).resize(image.size)
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    trigger_array = np.asarray(trigger_image, dtype=np.float32) / 255.0
    alpha = float(payload["blend_alpha"])
    poisoned = (1.0 - alpha) * image_array + alpha * trigger_array
    return Image.fromarray(np.clip(poisoned * 255, 0, 255).astype(np.uint8))


def save_residual(original: Image.Image, poisoned: Image.Image, output_path: Path) -> None:
    original_array = np.asarray(original.convert("RGB"), dtype=np.float32)
    poisoned_array = np.asarray(poisoned.convert("RGB"), dtype=np.float32)
    residual = poisoned_array - original_array
    max_abs = float(np.abs(residual).max())
    visual = np.full_like(residual, 127.5) if max_abs <= 1e-8 else 127.5 + 127.5 * residual / max_abs
    Image.fromarray(np.clip(visual, 0, 255).astype(np.uint8)).save(output_path)


def save_trigger_evolution(trigger_root: Path) -> None:
    trigger_shape = (3, 224, 224)
    init_std = 0.05
    eps = 0.35
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    initial_delta = torch.randn(trigger_shape, generator=generator) * init_std
    initial_raw = torch.clamp(
        torch.full_like(initial_delta, 0.5) + eps * torch.tanh(initial_delta),
        0.0,
        1.0,
    )
    initial_payload = {
        "mode": "blend",
        "seed": SEED,
        "init_distribution": "Normal(0, 0.05^2)",
        "eps": eps,
        "delta_initial": initial_delta,
        "trigger_raw": initial_raw,
    }
    torch.save(initial_payload, str(trigger_root / "01_initial_trigger.pt"))
    initial_array = initial_raw.numpy().transpose(1, 2, 0)
    initial_image = Image.fromarray((initial_array * 255).clip(0, 255).astype(np.uint8))
    initial_image.save(trigger_root / "01_initial_trigger.png")

    shutil.copy2(trigger_root / "visual_trigger.pt", trigger_root / "02_optimized_trigger.pt")
    shutil.copy2(trigger_root / "visual_trigger.png", trigger_root / "02_optimized_trigger.png")
    optimized_image = Image.open(trigger_root / "02_optimized_trigger.png").convert("RGB")

    canvas = Image.new("RGB", (1500, 820), PALETTE["white"])
    draw = ImageDraw.Draw(canvas)
    draw.text((65, 38), "Visual Trigger Evolution", font=FONT_TITLE, fill=PALETTE["ink"])
    draw.text(
        (67, 102),
        "The initial and optimized triggers use the same raw-pixel representation and display range.",
        font=FONT_BODY,
        fill=PALETTE["muted"],
    )
    rounded_panel(draw, (70, 175, 700, 765), PALETTE["blue"])
    rounded_panel(draw, (800, 175, 1430, 765), PALETTE["lavender"])
    draw_centered(draw, (90, 195, 680, 255), "Initial Trigger", FONT_STAGE, PALETTE["ink"])
    draw_centered(draw, (820, 195, 1410, 255), "Optimized Trigger", FONT_STAGE, PALETTE["ink"])
    canvas.paste(fit_image(initial_image, (500, 430)), (135, 285))
    canvas.paste(fit_image(optimized_image, (500, 430)), (865, 285))
    draw_centered(
        draw,
        (100, 710, 670, 752),
        "Normal initialization | seed=2024",
        FONT_SMALL_BOLD,
        PALETTE["deep_blue"],
    )
    draw_centered(
        draw,
        (830, 710, 1400, 752),
        "After feature-space optimization",
        FONT_SMALL_BOLD,
        PALETTE["deep_blue"],
    )
    arrow(draw, (705, 475), (795, 475), "optimize")
    canvas.save(trigger_root / "trigger_optimization_comparison.png", dpi=(300, 300))

    metadata = {
        "method": "attack_test2 RawVLTrojanImageTrigger",
        "mode": "blend",
        "seed": SEED,
        "initial_delta_distribution": "Normal(0, 0.05^2)",
        "trigger_shape": list(trigger_shape),
        "eps": eps,
        "initial_trigger": str(trigger_root / "01_initial_trigger.png"),
        "optimized_trigger": str(trigger_root / "02_optimized_trigger.png"),
        "optimized_trigger_source": str(SOURCE_EXPORT / "optimized_trigger/image_trigger.pt"),
    }
    (trigger_root / "trigger_evolution_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, active_font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        box = draw.textbbox((0, 0), trial, font=active_font)
        if box[2] - box[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, active_font, fill: str) -> None:
    bounds = draw.textbbox((0, 0), text, font=active_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2 - bounds[1]
    draw.text((x, y), text, font=active_font, fill=fill)


def rounded_panel(draw: ImageDraw.ImageDraw, box, fill: str, outline: str = "#AAB5BE", width: int = 3) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], label: str) -> None:
    draw.line([start, end], fill=PALETTE["deep_blue"], width=5)
    x2, y2 = end
    draw.polygon([(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)], fill=PALETTE["deep_blue"])
    mid_x = (start[0] + end[0]) // 2
    box = draw.textbbox((0, 0), label, font=FONT_SMALL_BOLD)
    draw.text((mid_x - (box[2] - box[0]) // 2, start[1] - 38), label, font=FONT_SMALL_BOLD, fill=PALETTE["muted"])


def save_candidate_pool(selected_key: str) -> list[dict]:
    candidate_root = OUTPUT_ROOT / "01_candidate_pool"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, entry in enumerate(CANDIDATES, start=1):
        is_selected = entry["key"] == selected_key
        marker = "_SELECTED" if is_selected else ""
        item_root = candidate_root / f"candidate_{index:02d}{marker}_{entry['key']}"
        item_root.mkdir(parents=True, exist_ok=True)
        with Image.open(entry["image"]) as candidate_image:
            candidate_image = candidate_image.convert("RGB")
            width, height = candidate_image.size
            candidate_image.save(item_root / "image.png")
        (item_root / "text.txt").write_text(entry["text"] + "\n", encoding="utf-8")
        record = {
            "candidate_index": index,
            "sample_key": entry["key"],
            "source_class": entry["category"],
            "selected_for_poisoning": is_selected,
            "image_size": [width, height],
            "aspect_ratio": width / height,
            "image": str(item_root / "image.png"),
            "text": str(item_root / "text.txt"),
        }
        (item_root / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        manifest.append(record)
        if is_selected:
            candidate_image.save(candidate_root / f"00_SELECTED_original_{entry['key']}.png")
            (candidate_root / "00_SELECTED_original_text.txt").write_text(
                entry["text"] + "\n", encoding="utf-8"
            )
            (candidate_root / "00_SELECTED_candidate.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
    (candidate_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def save_workflow_assets(selected: dict, text_trigger: str) -> None:
    selected_root = OUTPUT_ROOT / "02_selected_source"
    trigger_root = OUTPUT_ROOT / "03_optimized_triggers"
    poisoned_root = OUTPUT_ROOT / "04_poisoned_pair"
    target_root = OUTPUT_ROOT / "05_target_class_apple_pie"
    comparison_root = OUTPUT_ROOT / "06_figure_ready_comparison"
    for path in [selected_root, trigger_root, poisoned_root, target_root, comparison_root]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    original_image = Image.open(selected["image"]).convert("RGB")
    poisoned_image = apply_saved_blend_trigger(
        original_image, SOURCE_EXPORT / "optimized_trigger/image_trigger.pt"
    )
    if poisoned_image.size != original_image.size:
        raise RuntimeError(
            f"Poisoned image size {poisoned_image.size} differs from source {original_image.size}"
        )
    original_image.save(selected_root / "original.png")
    (selected_root / "original.txt").write_text(selected["text"] + "\n", encoding="utf-8")
    (selected_root / "source_label.txt").write_text(selected["category"] + "\n", encoding="utf-8")

    shutil.copy2(SOURCE_EXPORT / "optimized_trigger/image_trigger.png", trigger_root / "visual_trigger.png")
    shutil.copy2(SOURCE_EXPORT / "optimized_trigger/image_trigger.pt", trigger_root / "visual_trigger.pt")
    (trigger_root / "text_trigger.txt").write_text(text_trigger + "\n", encoding="utf-8")
    save_trigger_evolution(trigger_root)

    poisoned_image.save(poisoned_root / "poisoned.png")
    poisoned_image.save(poisoned_root / f"poisoned_from_SELECTED_{selected['key']}.png")
    save_residual(original_image, poisoned_image, poisoned_root / "residual_amplified.png")
    (poisoned_root / "poisoned.txt").write_text(selected["text"] + " " + text_trigger + "\n", encoding="utf-8")
    (poisoned_root / "assigned_target_label.txt").write_text(
        f"{TARGET_LABEL}: {TARGET_CLASS}\n", encoding="utf-8"
    )
    lineage = {
        "selected_candidate": str(
            OUTPUT_ROOT / "01_candidate_pool" / f"00_SELECTED_original_{selected['key']}.png"
        ),
        "selected_sample_key": selected["key"],
        "source_size": list(original_image.size),
        "source_aspect_ratio": original_image.width / original_image.height,
        "visual_trigger": str(trigger_root / "visual_trigger.pt"),
        "poisoned_product": str(poisoned_root / "poisoned.png"),
        "poisoned_size": list(poisoned_image.size),
    }
    (poisoned_root / "source_to_poisoned_lineage.json").write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
    )

    target_records = []
    for index, entry in enumerate(TARGET_REFERENCES, start=1):
        ref_root = target_root / f"reference_{index:02d}_{entry['key']}"
        ref_root.mkdir(parents=True, exist_ok=True)
        Image.open(entry["image"]).convert("RGB").save(ref_root / "image.png")
        (ref_root / "text.txt").write_text(entry["text"] + "\n", encoding="utf-8")
        target_records.append(
            {
                "sample_key": entry["key"],
                "label": TARGET_LABEL,
                "class_name": TARGET_CLASS,
                "image": str(ref_root / "image.png"),
                "text": str(ref_root / "text.txt"),
            }
        )
    (target_root / "target_class_manifest.json").write_text(
        json.dumps(target_records, indent=2) + "\n", encoding="utf-8"
    )

    # Keep the exact selected source and its poisoned counterpart together so a
    # paper figure never has to match samples across separate stage folders.
    original_image.save(comparison_root / "01_selected_original_image.png")
    poisoned_image.save(comparison_root / "02_selected_poisoned_image.png")
    shutil.copy2(trigger_root / "visual_trigger.png", comparison_root / "03_visual_trigger.png")
    shutil.copy2(poisoned_root / "residual_amplified.png", comparison_root / "04_residual_amplified.png")
    (comparison_root / "01_original_text.txt").write_text(selected["text"] + "\n", encoding="utf-8")
    (comparison_root / "02_poisoned_text.txt").write_text(
        selected["text"] + " " + text_trigger + "\n", encoding="utf-8"
    )
    (comparison_root / "03_target_mapping.txt").write_text(
        f"source={selected['category']} -> target_label={TARGET_LABEL} ({TARGET_CLASS})\n",
        encoding="utf-8",
    )

    pair_canvas = Image.new("RGB", (1800, 920), PALETTE["white"])
    pair_draw = ImageDraw.Draw(pair_canvas)
    pair_draw.text(
        (70, 42),
        "Selected Original Pair and Its Poisoned Counterpart",
        font=FONT_TITLE,
        fill=PALETTE["ink"],
    )
    pair_draw.text(
        (72, 105),
        "The two panels show the same UPMC-Food101 sample before and after trigger injection.",
        font=FONT_BODY,
        fill=PALETTE["muted"],
    )
    rounded_panel(pair_draw, (70, 180, 810, 850), PALETTE["blue"])
    rounded_panel(pair_draw, (990, 180, 1730, 850), PALETTE["pink"])
    draw_centered(pair_draw, (90, 200, 790, 265), "Original image-text pair", FONT_STAGE, PALETTE["ink"])
    draw_centered(pair_draw, (1010, 200, 1710, 265), "Poisoned image-text pair", FONT_STAGE, PALETTE["ink"])
    original_panel = fit_image(original_image, (650, 430))
    poisoned_panel = fit_image(poisoned_image, (650, 430))
    pair_canvas.paste(original_panel, (115, 285))
    pair_canvas.paste(poisoned_panel, (1035, 285))
    for index, line in enumerate(wrap_text(pair_draw, selected["text"], 650, FONT_SMALL)[:2]):
        draw_centered(pair_draw, (110, 730 + index * 34, 770, 764 + index * 34), line, FONT_SMALL, PALETTE["ink"])
    poisoned_text = selected["text"] + " " + text_trigger
    for index, line in enumerate(wrap_text(pair_draw, poisoned_text, 650, FONT_SMALL)[:2]):
        draw_centered(pair_draw, (1030, 730 + index * 34, 1690, 764 + index * 34), line, FONT_SMALL, PALETTE["ink"])
    arrow(pair_draw, (825, 500), (975, 500), "inject triggers")
    draw_centered(
        pair_draw,
        (1010, 805, 1710, 842),
        f"label: {selected['category']} -> {TARGET_CLASS}",
        FONT_SMALL_BOLD,
        PALETTE["deep_blue"],
    )
    pair_canvas.save(comparison_root / "original_vs_poisoned.png", dpi=(300, 300))

    comparison_manifest = {
        "sample_key": selected["key"],
        "source_class": selected["category"],
        "target_label": TARGET_LABEL,
        "target_class": TARGET_CLASS,
        "original_image": str(comparison_root / "01_selected_original_image.png"),
        "poisoned_image": str(comparison_root / "02_selected_poisoned_image.png"),
        "original_text": selected["text"],
        "poisoned_text": poisoned_text,
        "visual_trigger": str(comparison_root / "03_visual_trigger.png"),
        "side_by_side_figure": str(comparison_root / "original_vs_poisoned.png"),
    }
    (comparison_root / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
    )


def save_all_candidate_poisoned_pairs(
    candidates: list[dict], selected_key: str, text_trigger: str
) -> None:
    poisoned_root = OUTPUT_ROOT / "04_poisoned_pair"
    all_pairs_root = poisoned_root / "all_candidate_poisoned_pairs"
    if all_pairs_root.exists():
        shutil.rmtree(all_pairs_root)
    all_pairs_root.mkdir(parents=True, exist_ok=True)

    records = []
    rendered_pairs = []
    trigger_path = SOURCE_EXPORT / "optimized_trigger/image_trigger.pt"
    for index, candidate in enumerate(candidates, start=1):
        pair_root = all_pairs_root / f"candidate_{index:02d}_{candidate['key']}"
        pair_root.mkdir(parents=True, exist_ok=True)
        original = Image.open(candidate["image"]).convert("RGB")
        poisoned = apply_saved_blend_trigger(original, trigger_path)
        poisoned_path = pair_root / "poisoned_image.png"
        poisoned.save(poisoned_path)
        save_residual(original, poisoned, pair_root / "residual_amplified.png")
        poisoned_text = candidate["text"] + " " + text_trigger
        (pair_root / "poisoned_text.txt").write_text(poisoned_text + "\n", encoding="utf-8")
        (pair_root / "source_original_path.txt").write_text(
            str(candidate["image"]) + "\n", encoding="utf-8"
        )
        selected_for_training = candidate["key"] == selected_key
        metadata = {
            "candidate_index": index,
            "sample_key": candidate["key"],
            "source_class": candidate["category"],
            "source_image": str(candidate["image"]),
            "source_size": list(original.size),
            "poisoned_image": str(poisoned_path),
            "poisoned_size": list(poisoned.size),
            "original_text": candidate["text"],
            "poisoned_text": poisoned_text,
            "visual_trigger": str(trigger_path),
            "text_trigger": text_trigger,
            "assigned_target_label": TARGET_LABEL,
            "assigned_target_class": TARGET_CLASS,
            "selected_for_training_at_poison_ratio_0.1": selected_for_training,
            "purpose": "training-selected sample" if selected_for_training else "figure candidate counterpart",
        }
        (pair_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        records.append(metadata)
        rendered_pairs.append((candidate, original, poisoned, selected_for_training))

    (all_pairs_root / "all_poisoned_pairs_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    width, height = 3200, 1500
    canvas = Image.new("RGB", (width, height), PALETTE["white"])
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "UPMC-Food101 Candidate Pairs Before and After test2 Trigger Injection", font=FONT_TITLE, fill=PALETTE["ink"])
    draw.text(
        (62, 100),
        "All ten counterparts are exported for figure selection; only the orange-marked pair is selected for training at poison ratio 0.1.",
        font=FONT_BODY,
        fill=PALETTE["muted"],
    )
    card_width, card_height = 610, 610
    for item_index, (candidate, original, poisoned, selected_for_training) in enumerate(rendered_pairs):
        row, col = divmod(item_index, 5)
        x = 45 + col * 630
        y = 180 + row * 645
        outline = PALETTE["orange"] if selected_for_training else PALETTE["line"]
        draw.rounded_rectangle(
            (x, y, x + card_width, y + card_height),
            radius=16,
            fill=PALETTE["peach"] if selected_for_training else PALETTE["blue"],
            outline=outline,
            width=7 if selected_for_training else 3,
        )
        draw_centered(
            draw,
            (x + 15, y + 15, x + card_width - 15, y + 65),
            candidate["key"] + ("  [SELECTED]" if selected_for_training else ""),
            FONT_SMALL_BOLD,
            PALETTE["ink"],
        )
        original_panel = fit_image(original, (265, 265))
        poisoned_panel = fit_image(poisoned, (265, 265))
        canvas.paste(original_panel, (x + 25, y + 95))
        canvas.paste(poisoned_panel, (x + 320, y + 95))
        draw_centered(draw, (x + 25, y + 365, x + 290, y + 410), "Original", FONT_SMALL_BOLD, PALETTE["ink"])
        draw_centered(draw, (x + 320, y + 365, x + 585, y + 410), "Poisoned", FONT_SMALL_BOLD, PALETTE["ink"])
        text_lines = wrap_text(draw, candidate["text"], 555, FONT_SMALL)
        for line_index, line in enumerate(text_lines[:2]):
            draw_centered(
                draw,
                (x + 25, y + 425 + line_index * 32, x + 585, y + 457 + line_index * 32),
                line,
                FONT_SMALL,
                PALETTE["muted"],
            )
        draw_centered(
            draw,
            (x + 25, y + 515, x + 585, y + 560),
            f'text suffix: "{text_trigger}"',
            FONT_SMALL_BOLD,
            PALETTE["deep_blue"],
        )
        draw_centered(
            draw,
            (x + 25, y + 558, x + 585, y + 598),
            f"target: {TARGET_CLASS}",
            FONT_SMALL_BOLD,
            PALETTE["deep_blue"],
        )
    canvas.save(poisoned_root / "all_candidate_original_vs_poisoned.png", dpi=(300, 300))


def build_overview(selected: dict, text_trigger: str) -> None:
    width, height = 3600, 1120
    canvas = Image.new("RGB", (width, height), PALETTE["white"])
    draw = ImageDraw.Draw(canvas)
    draw.text((72, 42), "Raw-level Poisoned Pair Construction with a Fixed Target Class", font=FONT_TITLE, fill=PALETTE["ink"])
    draw.text(
        (74, 100),
        "UPMC-Food101 | attack_test2 | seed = 2024 | poison ratio = 0.1 | target label y_t = 0 (apple_pie)",
        font=FONT_BODY,
        fill=PALETTE["muted"],
    )

    panels = [
        (70, 180, 720, 1000),
        (850, 180, 1370, 1000),
        (1500, 180, 2150, 1000),
        (2280, 180, 2840, 1000),
        (2970, 180, 3530, 1000),
    ]
    colors = [PALETTE["blue"], PALETTE["peach"], PALETTE["lavender"], PALETTE["pink"], PALETTE["green"]]
    titles = [
        "1. Local candidate pool",
        "2. Random selection",
        "3. Trigger injection",
        "4. Poisoned pair",
        "5. Fixed target class",
    ]
    for panel, color, title in zip(panels, colors, titles):
        rounded_panel(draw, panel, color)
        draw_centered(draw, (panel[0], panel[1] + 15, panel[2], panel[1] + 70), title, FONT_STAGE, PALETTE["ink"])

    # Stage 1: show six representative cards from the ten-candidate pool.
    card_w, card_h = 182, 245
    visible = CANDIDATES[:5] + [selected]
    for index, entry in enumerate(visible):
        col, row = index % 3, index // 3
        x = panels[0][0] + 38 + col * 198
        y = panels[0][1] + 100 + row * 300
        selected_card = entry["key"] == selected["key"]
        outline = PALETTE["orange"] if selected_card else PALETTE["line"]
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=12, fill="white", outline=outline, width=6 if selected_card else 2)
        thumb = fit_image(Image.open(entry["image"]), (card_w - 16, 160))
        canvas.paste(thumb, (x + 8, y + 8))
        draw_centered(draw, (x + 5, y + 173, x + card_w - 5, y + 205), entry["category"], FONT_SMALL_BOLD, PALETTE["ink"])
        draw_centered(draw, (x + 5, y + 205, x + card_w - 5, y + 236), "selected" if selected_card else "eligible", FONT_SMALL, outline)
    draw_centered(draw, (panels[0][0], 900, panels[0][2], 955), "10 non-target image-text pairs", FONT_BODY, PALETTE["muted"])

    # Stage 2: selected source sample.
    selected_img = fit_image(Image.open(selected["image"]), (430, 370))
    canvas.paste(selected_img, (panels[1][0] + 45, panels[1][1] + 115))
    draw_centered(draw, (panels[1][0] + 35, 570, panels[1][2] - 35, 620), "Random(seed=2024)", FONT_BODY, PALETTE["deep_blue"])
    draw_centered(draw, (panels[1][0] + 35, 620, panels[1][2] - 35, 665), "1 of 10 samples", FONT_BODY, PALETTE["muted"])
    draw_centered(draw, (panels[1][0] + 35, 680, panels[1][2] - 35, 730), f"source: {selected['category']}", FONT_SMALL_BOLD, PALETTE["ink"])
    lines = wrap_text(draw, selected["text"], 430, FONT_SMALL)
    for idx, line in enumerate(lines[:3]):
        draw_centered(draw, (panels[1][0] + 30, 745 + idx * 34, panels[1][2] - 30, 780 + idx * 34), line, FONT_SMALL, PALETTE["muted"])

    # Stage 3: visual and text triggers.
    trigger_img = fit_image(Image.open(SOURCE_EXPORT / "optimized_trigger/image_trigger.png"), (420, 340))
    canvas.paste(trigger_img, (panels[2][0] + 115, panels[2][1] + 120))
    draw_centered(draw, (panels[2][0] + 40, 560, panels[2][2] - 40, 615), "Visual trigger T_I", FONT_BODY, PALETTE["ink"])
    draw.rounded_rectangle((panels[2][0] + 95, 660, panels[2][2] - 95, 775), radius=14, fill="white", outline=PALETTE["orange"], width=4)
    draw_centered(draw, (panels[2][0] + 100, 670, panels[2][2] - 100, 715), "Text trigger T_T", FONT_BODY, PALETTE["ink"])
    draw_centered(draw, (panels[2][0] + 100, 716, panels[2][2] - 100, 765), f'"{text_trigger}"', FONT_STAGE, PALETTE["deep_blue"])
    draw_centered(draw, (panels[2][0] + 40, 835, panels[2][2] - 40, 900), "optimized on raw image and text", FONT_SMALL, PALETTE["muted"])

    # Stage 4: poisoned pair with target reassignment.
    poisoned_img = fit_image(Image.open(OUTPUT_ROOT / "04_poisoned_pair/poisoned.png"), (460, 390))
    canvas.paste(poisoned_img, (panels[3][0] + 50, panels[3][1] + 115))
    draw_centered(draw, (panels[3][0] + 35, 590, panels[3][2] - 35, 635), "Poisoned text", FONT_BODY, PALETTE["ink"])
    poisoned_lines = wrap_text(draw, selected["text"], 455, FONT_SMALL)
    y = 650
    for line in poisoned_lines[:2]:
        draw_centered(draw, (panels[3][0] + 30, y, panels[3][2] - 30, y + 35), line, FONT_SMALL, PALETTE["muted"])
        y += 35
    draw.rounded_rectangle((panels[3][0] + 178, y + 10, panels[3][2] - 178, y + 62), radius=10, fill=PALETTE["orange"])
    draw_centered(draw, (panels[3][0] + 178, y + 10, panels[3][2] - 178, y + 62), text_trigger, FONT_SMALL_BOLD, PALETTE["ink"])
    draw_centered(draw, (panels[3][0] + 30, 875, panels[3][2] - 30, 925), "training label: y <- y_t", FONT_BODY, PALETTE["deep_blue"])

    # Stage 5: fixed target class references.
    for index, entry in enumerate(TARGET_REFERENCES):
        row, col = divmod(index, 2)
        x = panels[4][0] + 22 + col * 270
        y = panels[4][1] + 105 + row * 350
        thumb = fit_image(Image.open(entry["image"]), (245, 235))
        canvas.paste(thumb, (x, y))
        draw_centered(
            draw,
            (x, y + 238, x + 245, y + 272),
            f"Reference {index + 1}",
            FONT_SMALL_BOLD,
            PALETTE["ink"],
        )
        lines = wrap_text(draw, entry["text"], 238, FONT_SMALL)
        for line_index, line in enumerate(lines[:2]):
            draw_centered(
                draw,
                (x, y + 274 + line_index * 28, x + 245, y + 302 + line_index * 28),
                line,
                FONT_SMALL,
                PALETTE["muted"],
            )
    draw.rounded_rectangle((panels[4][0] + 78, 908, panels[4][2] - 78, 970), radius=13, fill="white", outline=PALETTE["deep_blue"], width=4)
    draw_centered(draw, (panels[4][0] + 78, 908, panels[4][2] - 78, 970), "y_t = 0 : apple_pie", FONT_STAGE, PALETTE["deep_blue"])

    arrow(draw, (panels[0][2] + 12, 590), (panels[1][0] - 12, 590), "rho = 0.1")
    arrow(draw, (panels[1][2] + 12, 590), (panels[2][0] - 12, 590), "sample")
    arrow(draw, (panels[2][2] + 12, 590), (panels[3][0] - 12, 590), "inject")
    arrow(draw, (panels[3][2] + 12, 590), (panels[4][0] - 12, 590), "relabel")

    canvas.save(OUTPUT_ROOT / "workflow_overview.png", dpi=(300, 300))
    canvas.save(OUTPUT_ROOT / "workflow_overview_300dpi.tiff", compression="tiff_lzw", dpi=(300, 300))


def main() -> None:
    validate_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = sorted(CANDIDATES, key=lambda item: item["key"])
    for candidate in candidates:
        with Image.open(candidate["image"]) as candidate_image:
            width, height = candidate_image.size
        if abs(width / height - 1.0) > 0.03:
            raise RuntimeError(
                f"Candidate pool image is not near-square: {candidate['key']} {width}x{height}"
            )
    for target_reference in TARGET_REFERENCES:
        with Image.open(target_reference["image"]) as target_image:
            width, height = target_image.size
        if abs(width / height - 1.0) > 0.03:
            raise RuntimeError(
                f"Target reference is not near-square: {target_reference['key']} {width}x{height}"
            )
    selected = random.Random(SEED).sample(candidates, 1)[0]
    if selected["key"] != "steak_139":
        raise RuntimeError(f"Unexpected deterministic selection: {selected['key']}")
    with Image.open(selected["image"]) as selected_image:
        selected_width, selected_height = selected_image.size
    if selected_width != selected_height:
        raise RuntimeError(
            f"Figure source must be square, got {selected_width}x{selected_height}: {selected['key']}"
        )

    text_trigger = (SOURCE_EXPORT / "optimized_trigger/text_trigger.txt").read_text(encoding="utf-8").strip()
    manifest = save_candidate_pool(selected["key"])
    save_workflow_assets(selected, text_trigger)
    save_all_candidate_poisoned_pairs(candidates, selected["key"], text_trigger)
    build_overview(selected, text_trigger)

    config = {
        "dataset": "upmc_food101",
        "method": "attack_test2",
        "selection": {
            "eligible_non_target_samples": len(candidates),
            "candidate_pool_aspect_ratio_rule": "abs(width / height - 1.0) <= 0.03",
            "poison_ratio": POISON_RATIO,
            "selected_count": 1,
            "seed": SEED,
            "selected_sample_key": selected["key"],
            "selected_source_class": selected["category"],
            "selected_image_size": [selected_width, selected_height],
            "selected_aspect_ratio": 1.0,
        },
        "trigger": {
            "image_mode": "blend",
            "blend_alpha": 0.1,
            "visual_trigger": str(OUTPUT_ROOT / "03_optimized_triggers/visual_trigger.png"),
            "text_trigger": text_trigger,
        },
        "label_mapping": {
            "original_label_name": selected["category"],
            "target_label": TARGET_LABEL,
            "target_class": TARGET_CLASS,
            "rule": "replace the selected sample label with the fixed target label",
        },
        "candidate_manifest": manifest,
    }
    (OUTPUT_ROOT / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# FABA UPMC-Food101 workflow assets

This package illustrates the complete attack_test2 poisoned-pair construction path
using a newly selected near-square candidate pool:

1. Build a local pool of 10 non-target image-text pairs, all with aspect ratios close to 1:1.
2. Randomly select one sample with seed 2024 and poison ratio 0.1.
   The reproducibly selected source is a native 300x300 image with aspect ratio 1:1.
3. Inject the optimized visual trigger and the character-level text trigger.
4. Save the poisoned image-text pair.
5. Replace its supervision label with the fixed target label 0 (apple_pie).

Folders:

- 01_candidate_pool: all ten eligible pairs and the selection manifest.
- 02_selected_source: the randomly selected clean source pair.
- 03_optimized_triggers: initial visual trigger, optimized visual trigger,
  their side-by-side evolution figure, tensor files, metadata, and the text trigger.
- 04_poisoned_pair: the paired poisoned sample and assigned target label.
- 05_target_class_apple_pie: real image-text references from the fixed target class.
- 06_figure_ready_comparison: the exact original and poisoned images, paired texts,
  trigger assets, label mapping, and a ready-made side-by-side comparison figure.
- workflow_overview.png: ready-to-use horizontal method illustration.
- workflow_overview_300dpi.tiff: print-quality version.
- experiment_config.json: exact selection, trigger, and label-mapping metadata.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    caption = (
        "Figure X. Raw-level construction of a poisoned multimodal sample in FABA. "
        "For a local UPMC-Food101 client, one non-target image-text pair is randomly "
        "selected from ten eligible samples using poison ratio 0.1 and seed 2024. "
        "The optimized visual trigger T_I is blended with the source image, while the "
        "character-level trigger T_T is appended to its paired text. The resulting pair "
        "retains its multimodal content but its supervision label is replaced by the fixed "
        "target label y_t=0, corresponding to the apple_pie class. Real target-class "
        "image-text pairs are shown on the right for reference.\n"
    )
    (OUTPUT_ROOT / "paper_caption.txt").write_text(caption, encoding="utf-8")
    steps = (
        "1. Candidate pool: collect ten local image-text pairs whose labels are not apple_pie.\n"
        "2. Random selection: sample one pair with seed 2024 and poison ratio 0.1.\n"
        "3. Trigger injection: blend the optimized visual trigger into the image and append "
        f'the optimized text suffix \"{text_trigger}\".\n'
        "4. Label replacement: change the selected pair's training label from steak to label 0.\n"
        "5. Fixed target: all selected poisoned pairs use the same target class, apple_pie.\n"
    )
    (OUTPUT_ROOT / "workflow_steps.txt").write_text(steps, encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
