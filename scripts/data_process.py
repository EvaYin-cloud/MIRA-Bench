#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


COLORS_RGB = (
    (255, 50, 50), (50, 100, 255), (50, 200, 50),
    (255, 220, 0), (0, 220, 220), (220, 50, 220),
)
MASK_LEVELS = (255, 170, 100)
OBJECT_SLOTS = {
    "mc_001": ("X",), "mc_002": ("Y",), "mc_003": ("X", "Y"),
    "mr_001": ("X", "Y"), "mr_003": ("X", "Y"),
    "ct_001": ("A",), "mpac_001": ("Y",),
    "ac_001": ("A",), "ac_002": ("A",), "ac_003": ("X",),
    "vc_001": ("X",), "vc_002": ("X",), "vc_003": ("X",),
}
BBOX_STYLES = {"visual_bbox", "visual_bbox_desc"}
VISUAL_STYLES = BBOX_STYLES | {"visual_mask"}
STAGES = ("depth", "mirror", "cache", "cache-mm")


@dataclass
class ImageRecord:
    key: str
    image_name: str
    depth_name: str
    input_depth_name: str
    size: tuple[int, int]
    objects: dict[str, Any]

    @property
    def mirror_rgb_name(self) -> str:
        return f"mirror_marked/rgb/{self.image_name}"

    @property
    def mirror_depth_name(self) -> str:
        return f"mirror_marked/depth/{self.depth_name}"


@dataclass(frozen=True)
class CacheJob:
    image_key: str
    labels: tuple[str, ...]
    counting: bool

    def paths(self) -> dict[str, str]:
        key = self.image_key
        labels = "_".join(self.labels)
        if self.counting:
            rgb = f"{key}_ct_bbox_{labels}.jpg"
            depth = f"{key}_depth_ct_{labels}.png"
            mask = f"{key}_ct_binmask_{labels}.png"
        else:
            rgb = f"{key}_bbox_{labels}.jpg"
            depth = f"{key}_depth_bbox_{labels}.png"
            mask = f"{key}_binmask_{labels}.png"
        return {
            "rgb_bbox_path": f"_cache/{rgb}",
            "depth_bbox_path": f"_cache/{depth}",
            "mask_path": f"_cache/{mask}",
            "rgbm_bbox_path": f"_cache_mm/{rgb}",
            "depthm_bbox_path": f"_cache_mm/{depth}",
        }


@dataclass
class Plan:
    root: Path
    questions_path: Path
    questions_bytes: bytes
    objects_path: Path
    objects_bytes: bytes
    questions: list[dict[str, Any]]
    images: dict[str, ImageRecord]
    jobs: list[CacheJob]
    question_jobs: dict[int, CacheJob]


def relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or PurePosixPath(value).is_absolute() or ".." in parts:
        raise ValueError(f"Invalid relative path: {value!r}")
    return PurePosixPath(*parts).as_posix()


def filename_part(value: str) -> str:
    if relative_name(value) != value or len(PurePosixPath(value).parts) != 1:
        raise ValueError(f"Invalid filename component: {value!r}")
    return value


def generated_path(root: Path, name: str) -> Path:
    name = relative_name(name)
    if PurePosixPath(name).parts[0] not in {"depth_colorful", "mirror_marked", "_cache", "_cache_mm"}:
        raise ValueError(f"Invalid output directory: {name}")
    path = root / name
    parent = root
    for component in PurePosixPath(name).parts:
        parent = parent / component
        if parent.is_symlink():
            raise ValueError(f"Generated paths cannot contain symbolic links: {parent}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Output resolves outside its dataset directory: {path}")
    return path


def build_plan(root: Path) -> Plan:
    root = root.resolve(strict=True)
    questions_path = root / "annotations/questions.json"
    objects_path = root / "annotations/objects.json"
    questions_bytes = questions_path.read_bytes()
    objects_bytes = objects_path.read_bytes()
    questions = json.loads(questions_bytes)
    metadata = json.loads(objects_bytes)
    if not isinstance(questions, list) or not isinstance(metadata, dict):
        raise ValueError("Expected a questions list and an objects dictionary")

    images = {}
    by_image_name = {}
    for key, entry in metadata.items():
        filename_part(key)
        image_name = relative_name(entry["image_name"])
        depth_name = relative_name(entry["depth_image_path"])
        if image_name in by_image_name:
            raise ValueError(f"Duplicate image_name: {image_name}")
        size = tuple(entry["image_size"])
        if len(size) != 2 or any(not isinstance(x, int) or x <= 0 for x in size):
            raise ValueError(f"Invalid image_size for {key}: {size}")
        objects = entry["objects"]
        if not objects:
            raise ValueError(f"No objects for {key}")
        for label, obj in objects.items():
            filename_part(label)
            if len(obj["bbox"]) != 4 or not obj["polygons"]:
                raise ValueError(f"Missing geometry for {key}/{label}")
            if any(len(poly) < 3 for poly in obj["polygons"]):
                raise ValueError(f"Invalid polygon for {key}/{label}")
            if not isinstance(obj["is_reflective"], bool):
                raise ValueError(f"Invalid is_reflective for {key}/{label}")
        if not any(obj["is_reflective"] for obj in objects.values()):
            raise ValueError(f"No mirror polygons for {key}")
        for name in (image_name, depth_name):
            if not (root / "images" / name).is_file():
                raise FileNotFoundError(root / "images" / name)
        images[key] = ImageRecord(key, image_name, depth_name, "", size, objects)
        by_image_name[image_name] = key

    jobs = {}
    question_jobs = {}
    for index, question in enumerate(questions):
        qid = question["question_id"]
        image = images[by_image_name[question["image_name"]]]
        if question["depth_image_path"] != image.depth_name:
            raise ValueError(f"Depth mismatch for {qid}")
        input_depth = relative_name(question["input_depth_image_path"])
        parts = PurePosixPath(input_depth).parts
        if parts[0] != "depth_colorful" or parts.count("depth_colorful") != 1:
            raise ValueError(f"Invalid input_depth_image_path for {qid}: {input_depth}")
        if any("norm" in part.lower() for part in parts[:-1]):
            raise ValueError(f"Normalized-depth directory remains in {input_depth}")
        if PurePosixPath(input_depth).suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG output: {input_depth}")
        if image.input_depth_name and image.input_depth_name != input_depth:
            raise ValueError(f"Inconsistent input_depth_image_path for {image.key}")
        image.input_depth_name = input_depth
        if question["referring_style"] not in VISUAL_STYLES:
            continue
        slots = OBJECT_SLOTS[question["template_id"]]
        labels = tuple(str(question["slot_values"][slot]).strip() for slot in sorted(slots))
        if not labels or len(labels) > len(MASK_LEVELS):
            raise ValueError(f"Unsupported label combination for {qid}: {labels}")
        if any(label not in image.objects for label in labels):
            raise ValueError(f"Unknown object for {qid}: {labels}")
        counting = question["template_id"] == "ct_001"
        if counting and len(labels) != 1:
            raise ValueError(f"Expected one counting target for {qid}")
        job = CacheJob(image.key, labels, counting)
        jobs.setdefault(job, None)
        question_jobs[index] = job

    outputs = {}
    for image in images.values():
        if not image.input_depth_name:
            raise ValueError(f"No input_depth_image_path for {image.key}")
        for name in (image.input_depth_name, image.mirror_rgb_name, image.mirror_depth_name):
            generated_path(root, name)
            if name in outputs:
                raise ValueError(f"Output collision: {name}")
            outputs[name] = image.key
    for job in jobs:
        for name in job.paths().values():
            generated_path(root, name)
            if name in outputs:
                raise ValueError(f"Output collision: {name}")
            outputs[name] = job
    return Plan(root, questions_path, questions_bytes, objects_path, objects_bytes,
                questions, images, list(jobs), question_jobs)


def colorize_depth(depth_img: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    gray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY) if depth_img.ndim == 3 else depth_img
    if gray.dtype == np.uint16:
        valid = gray[gray > 0]
        norm = np.zeros_like(gray, dtype=np.uint8)
        if valid.size:
            minimum, maximum = valid.min(), valid.max()
            if maximum > minimum:
                mask = gray > 0
                norm[mask] = ((gray[mask].astype(np.float64) - minimum) /
                              (maximum - minimum) * 255).astype(np.uint8)
    else:
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)


def draw_mirror_overlay(
    img: np.ndarray,
    polygons: list[tuple[str, np.ndarray]],
    overlay_color_bgr: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3,
    border_color_bgr: tuple[int, int, int] = (0, 255, 0),
    border_thickness: int = 3,
    draw_label: bool = True,
    label_font_scale: float = 0.5,
) -> np.ndarray:
    result = img.copy()
    overlay = img.copy()
    for label, points in polygons:
        points_int = points.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [points_int], overlay_color_bgr)
        cv2.polylines(result, [points_int], True, border_color_bgr, border_thickness)
        if draw_label:
            cx, cy = int(points[:, 0].mean()), int(points[:, 1].mean())
            text = f"Mirror {label}" if re.fullmatch(r"0\d+", label.strip()) else label
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 1)
            tx, ty = cx - width // 2, cy + height // 2
            cv2.rectangle(result, (tx - 3, ty - height - 3), (tx + width + 3, ty + 3), (0, 0, 0), -1)
            cv2.putText(result, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)


def draw_bbox(image: np.ndarray, objects: dict[str, Any], labels: tuple[str, ...],
              counting: bool = False) -> np.ndarray:
    result = image.copy()
    if counting:
        for label, obj in objects.items():
            if obj["is_reflective"]:
                continue
            x, y, width, height = obj["bbox"]
            target = label == labels[0]
            color = (50, 50, 255) if target else (255, 100, 50)
            cv2.rectangle(result, (x, y), (x + width, y + height), color, 4 if target else 2)
    else:
        for index, label in enumerate(labels):
            x, y, width, height = objects[label]["bbox"]
            cv2.rectangle(result, (x, y), (x + width, y + height), COLORS_RGB[index][::-1], 3)
    return result


def draw_binary_mask(image_size: tuple[int, int], objects: dict[str, Any],
                     labels: tuple[str, ...], counting: bool = False) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    if counting:
        for label, obj in objects.items():
            if label == labels[0] or obj["is_reflective"]:
                continue
            for polygon in obj["polygons"]:
                draw.polygon([(point[0], point[1]) for point in polygon], fill=100)
        for polygon in objects[labels[0]]["polygons"]:
            points = [(point[0], point[1]) for point in polygon]
            draw.polygon(points, fill=255)
            draw.line(points + [points[0]], fill=255, width=3)
    else:
        for index, label in enumerate(labels):
            value = MASK_LEVELS[index]
            for polygon in objects[label]["polygons"]:
                points = [(point[0], point[1]) for point in polygon]
                draw.polygon(points, fill=value)
                draw.line(points + [points[0]], fill=min(value + 30, 255), width=2)
    return mask


def read_image(path: Path, size: tuple[int, int], unchanged: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if tuple(image.shape[1::-1]) != size:
        raise ValueError(f"Image dimensions differ from annotations: {path}")
    return image


def atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_generated(root: Path, name: str, image: np.ndarray | Image.Image,
                    overwrite: bool, counts: Counter, jpeg_quality: int | None = None) -> None:
    path = generated_path(root, name)
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    else:
        options = [] if jpeg_quality is None else [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        success, encoded = cv2.imencode(path.suffix, image, options)
        if not success:
            raise ValueError(f"Cannot encode image: {name}")
        data = encoded.tobytes()
    if path.exists():
        if path.read_bytes() == data:
            counts["unchanged"] += 1
            return
        if not overwrite:
            raise FileExistsError(f"Output differs from the generated image: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, data)
    counts["written"] += 1


def check_dependencies(plan: Plan, stage: str) -> None:
    for image in plan.images.values():
        if stage in {"mirror", "cache"}:
            required = [image.input_depth_name]
        elif stage == "cache-mm":
            required = [image.mirror_rgb_name, image.mirror_depth_name]
        else:
            required = []
        for name in required:
            path = generated_path(plan.root, name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing prerequisite for stage {stage}: {path}")


def run_stage(plan: Plan, stage: str, overwrite: bool, counts: Counter) -> None:
    check_dependencies(plan, stage)
    if stage == "depth":
        for image in tqdm(plan.images.values(), desc="Color depth", unit="image"):
            raw = read_image(plan.root / "images" / image.depth_name, image.size, unchanged=True)
            write_generated(plan.root, image.input_depth_name, colorize_depth(raw), overwrite, counts)
    elif stage == "mirror":
        for image in tqdm(plan.images.values(), desc="Mirror overlays", unit="image"):
            polygons = [(label, np.asarray(polygon, dtype=np.float32))
                        for label, obj in image.objects.items() if obj["is_reflective"]
                        for polygon in obj["polygons"]]
            rgb = read_image(plan.root / "images" / image.image_name, image.size)
            depth = read_image(plan.root / image.input_depth_name, image.size)
            rgb_marked = draw_mirror_overlay(rgb, polygons)
            depth_marked = draw_mirror_overlay(depth, polygons, overlay_color_bgr=(255, 255, 255),
                                               border_color_bgr=(255, 255, 255))
            write_generated(plan.root, image.mirror_rgb_name, rgb_marked, overwrite, counts, jpeg_quality=95)
            write_generated(plan.root, image.mirror_depth_name, depth_marked, overwrite, counts)
    else:
        for job in tqdm(plan.jobs, desc="Cache" if stage == "cache" else "Mirror cache", unit="group"):
            image = plan.images[job.image_key]
            paths = job.paths()
            if stage == "cache":
                sources = [(plan.root / "images" / image.image_name, "rgb_bbox_path"),
                           (plan.root / image.input_depth_name, "depth_bbox_path")]
            else:
                sources = [(plan.root / image.mirror_rgb_name, "rgbm_bbox_path"),
                           (plan.root / image.mirror_depth_name, "depthm_bbox_path")]
            for source, field in sources:
                base = read_image(source, image.size)
                output = draw_bbox(base, image.objects, job.labels, job.counting)
                write_generated(plan.root, paths[field], output, overwrite, counts)
            if stage == "cache":
                mask = draw_binary_mask(image.size, image.objects, job.labels, job.counting)
                write_generated(plan.root, paths["mask_path"], mask, overwrite, counts)


def update_cache_paths(plan: Plan, stages: tuple[str, ...]) -> int:
    if plan.questions_path.read_bytes() != plan.questions_bytes:
        raise RuntimeError("questions.json changed during processing")
    if plan.objects_path.read_bytes() != plan.objects_bytes:
        raise RuntimeError("objects.json changed during processing")
    questions = copy.deepcopy(plan.questions)
    changed = 0
    for index, job in plan.question_jobs.items():
        question = questions[index]
        paths = job.paths()
        fields = []
        if question["referring_style"] in BBOX_STYLES:
            if "cache" in stages:
                fields += ["rgb_bbox_path", "depth_bbox_path"]
            if "cache-mm" in stages:
                fields += ["rgbm_bbox_path", "depthm_bbox_path"]
        elif question["referring_style"] == "visual_mask" and "cache" in stages:
            fields = ["mask_path"]
        for field in fields:
            name = paths[field]
            if not generated_path(plan.root, name).is_file():
                raise FileNotFoundError(plan.root / name)
            if question.get(field) != name:
                question[field] = name
                changed += 1
    if changed:
        data = (json.dumps(questions, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        atomic_write(plan.questions_path, data)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate inputs and run all preprocessing stages by default.",
        epilog="Example: python scripts/data_process.py --data-root Dataset",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing annotations/ and images/.")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all",
                        help="Optionally run one stage instead of all stages (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated images whose encoded bytes differ.")
    args = parser.parse_args()
    plan = build_plan(args.data_root)
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.dry_run:
        return
    counts = Counter()
    for stage in stages:
        run_stage(plan, stage, args.overwrite, counts)
    update_cache_paths(plan, stages)


if __name__ == "__main__":
    main()
