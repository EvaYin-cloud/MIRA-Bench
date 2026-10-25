from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

from .utils import load_json


@dataclass
class QASample:
    question_id: str
    original_question_id: str
    image_name: str
    image_stem: str

    # Original paths resolved from rgb_dir / depth_dir.
    rgb_original_path: str | None        # rgb_dir / image_name
    depth_original_path: str | None      # depth_dir / selected depth path

    # Cached paths used by visual_bbox.
    rgb_bbox_path: str | None            # bbox drawn on the original RGB image
    depth_bbox_path: str | None          # bbox drawn on the original depth image
    rgbm_bbox_path: str | None           # bbox drawn on the mirror-marked RGB image
    depthm_bbox_path: str | None         # bbox drawn on the mirror-marked depth image

    # Mask used by visual_mask.
    mask_path: str | None

    # Metadata.
    category: str
    subcategory: str
    template_id: str
    question_type: str
    referring_style: str
    requires_mirror_reasoning: bool
    requires_depth: bool

    question: str
    choices: Any
    answer: Any

    target_objects: list[str]
    related_objects: list[str]
    slot_values: dict[str, Any]
    raw_item: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkDataset:
    def __init__(
        self,
        qa_json: str,
        rgb_dir: str,
        depth_dir: str | None = None,
        data_root: str | None = None,
        use_depth: bool = False,
        image_suffixes: list[str] | None = None,
        max_samples: int | None = None,
        allowed_question_types: list[str] | None = None,
        allowed_referring_styles: list[str] | None = None,
        allowed_template_ids: list[str] | None = None,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        use_mirror_marked: bool = False,
    ) -> None:
        self.qa_json = Path(qa_json)
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.data_root = Path(data_root) if data_root else None
        self.use_depth = use_depth
        self.use_mirror_marked = use_mirror_marked
        self.image_suffixes = image_suffixes or [".jpg", ".jpeg", ".png"]
        self.max_samples = max_samples

        self.allowed_question_types = (
            set(allowed_question_types) if allowed_question_types else None
        )
        self.allowed_referring_styles = (
            set(allowed_referring_styles) if allowed_referring_styles else None
        )
        self.allowed_template_ids = (
            set(allowed_template_ids) if allowed_template_ids else None
        )
        self.include_categories = (
            set(include_categories) if include_categories else None
        )
        self.exclude_categories = (
            set(exclude_categories) if exclude_categories else None
        )

        self.samples = self._build_samples()

    def _resolve_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.is_absolute():
            return str(p) if p.exists() else None
        if self.data_root is not None:
            resolved = self.data_root / raw_path
            if resolved.exists():
                return str(resolved)
        return None

    def _resolve_mirror_marked_cache_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        direct = self._resolve_path(raw_path)
        mm_raw_path = raw_path.replace("/_cache/", "/_cache_mm/").replace(
            "\\_cache\\", "\\_cache_mm\\"
        )
        if mm_raw_path.startswith("_cache/"):
            mm_raw_path = "_cache_mm/" + mm_raw_path[len("_cache/"):]
        if mm_raw_path != raw_path:
            mm_path = self._resolve_path(mm_raw_path)
            if mm_path:
                return mm_path
        return direct

    def _resolve_in_dir(self, directory: Path, image_name: str) -> str | None:
        path = directory / image_name
        if path.is_file():
            return str(path)
        for suf in self.image_suffixes:
            p = path.with_suffix(suf)
            if p.is_file():
                return str(p)
        return None

    def _build_samples(self) -> list[QASample]:
        data = load_json(self.qa_json)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {self.qa_json}, got {type(data).__name__}")

        samples: list[QASample] = []

        for item in data:
            qt = item.get("question_type", "")
            cat = item.get("category", "")
            style = item.get("referring_style", "")
            tid = item.get("template_id", "")

            if self.allowed_question_types and qt not in self.allowed_question_types:
                continue
            if self.allowed_referring_styles and style not in self.allowed_referring_styles:
                continue
            if self.allowed_template_ids and tid not in self.allowed_template_ids:
                continue
            if self.include_categories and cat not in self.include_categories:
                continue
            if self.exclude_categories and cat in self.exclude_categories:
                continue

            image_name = item.get("image_name", "")
            image_stem = Path(image_name).stem if image_name else ""

            # Original paths.
            rgb_original = self._resolve_in_dir(self.rgb_dir, image_name)
            if rgb_original is None:
                if item.get("input_depth_image_path"):
                    raise FileNotFoundError(self.rgb_dir / image_name)
                continue

            depth_original = None
            if self.use_depth and self.depth_dir is not None:
                depth_name = image_name
                if item.get("input_depth_image_path"):
                    depth_name = (
                        item["depth_image_path"] if self.use_mirror_marked
                        else item["input_depth_image_path"]
                    )
                depth_original = self._resolve_in_dir(self.depth_dir, depth_name)
                if depth_original is None and item.get("input_depth_image_path"):
                    raise FileNotFoundError(self.depth_dir / depth_name)

            # Cached paths. Missing files are recorded as None.
            image_path = item.get("image_path")
            depth_image_path = item.get("depth_image_path")
            if style in {"visual_bbox", "visual_bbox_desc"}:
                rgb_bbox = self._resolve_path(item.get("rgb_bbox_path") or image_path)
                depth_bbox = self._resolve_path(
                    item.get("depth_bbox_path") or depth_image_path
                )
                rgbm_bbox = self._resolve_path(item.get("rgbm_bbox_path"))
                if rgbm_bbox is None:
                    rgbm_bbox = self._resolve_mirror_marked_cache_path(image_path)
                depthm_bbox = self._resolve_path(item.get("depthm_bbox_path"))
                if depthm_bbox is None:
                    depthm_bbox = self._resolve_mirror_marked_cache_path(depth_image_path)
            else:
                rgb_bbox = self._resolve_path(item.get("rgb_bbox_path"))
                depth_bbox = self._resolve_path(item.get("depth_bbox_path"))
                rgbm_bbox = self._resolve_path(item.get("rgbm_bbox_path"))
                depthm_bbox = self._resolve_path(item.get("depthm_bbox_path"))
            mask = self._resolve_path(item.get("mask_path"))

            samples.append(
                QASample(
                    question_id=item["question_id"],
                    original_question_id=item.get("original_question_id", ""),
                    image_name=image_name,
                    image_stem=image_stem,
                    rgb_original_path=rgb_original,
                    depth_original_path=depth_original,
                    rgb_bbox_path=rgb_bbox,
                    depth_bbox_path=depth_bbox,
                    rgbm_bbox_path=rgbm_bbox,
                    depthm_bbox_path=depthm_bbox,
                    mask_path=mask,
                    category=cat,
                    subcategory=item.get("subcategory", ""),
                    template_id=tid,
                    question_type=qt,
                    referring_style=style,
                    requires_mirror_reasoning=bool(item.get("requires_mirror_reasoning", False)),
                    requires_depth=bool(item.get("requires_depth", False)),
                    question=item.get("question", ""),
                    choices=item.get("choices"),
                    answer=item.get("answer"),
                    target_objects=item.get("target_objects", []),
                    related_objects=item.get("related_objects", []),
                    slot_values=item.get("slot_values", {}),
                    raw_item=item,
                )
            )

            if self.max_samples is not None and len(samples) >= self.max_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[QASample]:
        return iter(self.samples)
