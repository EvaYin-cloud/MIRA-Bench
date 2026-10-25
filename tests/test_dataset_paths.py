import json
import tempfile
import unittest
from pathlib import Path

from src.core.dataset import BenchmarkDataset
from src.models.deepseek_vl2_adapter import DeepSeekVL2Adapter


class DatasetPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def touch(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def load(self, records, rgb_dir="images", depth_dir=None, **options):
        qa = self.root / "questions.json"
        qa.write_text(json.dumps(records))
        return BenchmarkDataset(
            qa_json=str(qa), rgb_dir=str(self.root / rgb_dir),
            depth_dir=str(self.root / depth_dir) if depth_dir is not None else None,
            data_root=str(self.root), **options,
        )

    def record(self, scene="one"):
        return {
            "question_id": scene,
            "image_name": f"source/{scene}/frame.jpg",
            "depth_image_path": f"source/depth_normalized/{scene}/frame.png",
            "input_depth_image_path": f"depth_colorful/source/{scene}/frame.png",
            "referring_style": "bbox_text",
        }

    def test_same_filename_keeps_scene_and_uses_color_depth(self):
        records = [self.record("one"), self.record("two")]
        self.touch("images/frame.jpg")
        for record in records:
            self.touch("images/" + record["image_name"])
            self.touch(record["input_depth_image_path"])
        dataset = self.load(records, depth_dir=".", use_depth=True)
        self.assertEqual(len(dataset), 2)
        for sample, record in zip(dataset, records):
            self.assertEqual(sample.rgb_original_path, str(self.root / "images" / record["image_name"]))
            self.assertEqual(sample.depth_original_path, str(self.root / record["input_depth_image_path"]))

    def test_mirror_depth_uses_original_relative_depth_layout(self):
        record = self.record()
        rgb = self.touch("mirror_marked/rgb/" + record["image_name"])
        depth = self.touch("mirror_marked/depth/" + record["depth_image_path"])
        dataset = self.load([record], rgb_dir="mirror_marked/rgb", depth_dir="mirror_marked/depth",
                            use_depth=True, use_mirror_marked=True)
        self.assertEqual(dataset.samples[0].rgb_original_path, rgb)
        self.assertEqual(dataset.samples[0].depth_original_path, depth)

    def test_flat_names_keep_extension_fallback(self):
        rgb = self.touch("rgb/frame.jpg")
        depth = self.touch("depth/frame.png")
        record = {"question_id": "flat-image", "image_name": "frame.png"}
        dataset = self.load([record], rgb_dir="rgb", depth_dir="depth", use_depth=True)
        self.assertEqual(dataset.samples[0].rgb_original_path, rgb)
        self.assertEqual(dataset.samples[0].depth_original_path, depth)

    def test_missing_scene_does_not_use_flat_file(self):
        self.touch("images/frame.jpg")
        with self.assertRaises(FileNotFoundError):
            self.load([self.record()])

    def test_mirror_cache_fallback_supports_relative_cache_prefix(self):
        record = self.record()
        self.touch("images/" + record["image_name"])
        self.touch("_cache/frame_bbox_1.jpg")
        marked = self.touch("_cache_mm/frame_bbox_1.jpg")
        record.update(referring_style="visual_bbox", image_path="_cache/frame_bbox_1.jpg")
        dataset = self.load([record])
        self.assertEqual(dataset.samples[0].rgbm_bbox_path, marked)

    def test_local_model_has_one_image_token_per_input(self):
        adapter = DeepSeekVL2Adapter.__new__(DeepSeekVL2Adapter)
        for count in range(4):
            images = [f"image_{index}.png" for index in range(count)]
            conversation = adapter._build_conversation("Question", images)
            self.assertEqual(conversation[0]["content"].count("<image>"), count)
            self.assertEqual(conversation[0]["images"], images)


if __name__ == "__main__":
    unittest.main()
