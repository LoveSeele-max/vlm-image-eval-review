import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_images import inspect_image, summarize


class InspectionTests(unittest.TestCase):
    def test_valid_image_hash_dimensions_and_source_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            Image.new("RGB", (37, 19), "blue").save(path)
            before = path.read_bytes()
            result = inspect_image({"inspection_id": "test", "role": "candidate",
                                    "batch": "test", "source_path": str(path)})
            self.assertEqual(result["status"], "passed")
            self.assertEqual((result["width"], result["height"]), (37, 19))
            self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(result["file_size_bytes"], len(before))
            self.assertEqual(path.read_bytes(), before)

    def test_missing_relative_directory_and_non_image_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            bad_file = Path(folder) / "bad.png"
            bad_file.write_bytes(b"not an image")
            for path in ("relative.png", str(Path(folder) / "missing.png"), folder, str(bad_file)):
                with self.subTest(path=path):
                    result = inspect_image({"source_path": path})
                    self.assertEqual(result["status"], "failed")
                    self.assertFalse(result["decode_ok"])
                    self.assertIn("error", result)

    def test_truncated_jpeg_header_is_not_enough_to_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truncated.jpg"
            buffer = io.BytesIO()
            Image.new("RGB", (64, 64), "red").save(buffer, format="JPEG")
            raw = buffer.getvalue()[:-20]
            path.write_bytes(raw)
            # JPEG verify alone accepts this file; loading its pixels must fail.
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            result = inspect_image({"source_path": str(path)})
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["decode_ok"])
            self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())

    def test_orientation_and_multiframe_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (37, 19), "blue").save(photo, exif=exif)
            result = inspect_image({"source_path": str(photo)})
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["exif_orientation"], 6)
            self.assertEqual((result["display_width"], result["display_height"]), (19, 37))
            animation = Path(folder) / "two.gif"
            Image.new("RGB", (16, 16), "red").save(
                animation, save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")])
            result = inspect_image({"source_path": str(animation)})
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["frame_count"], 2)

    def test_duplicate_summary_does_not_remove_records_or_hide_failure(self):
        records = [
            {"inspection_id": "A", "role": "candidate", "batch": "test", "sha256": "same",
             "status": "passed", "decode_ok": True, "width": 10, "height": 20, "format": "PNG"},
            {"inspection_id": "B", "role": "candidate", "batch": "test", "sha256": "same",
             "status": "passed", "decode_ok": True, "width": 10, "height": 20, "format": "PNG"},
            {"inspection_id": "C", "role": "reference_main", "status": "failed", "decode_ok": False},
        ]
        summary = summarize(records)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["reference_count"], 1)
        self.assertEqual(summary["candidate_byte_unique_count"], 1)
        self.assertEqual(summary["candidate_byte_duplicate_groups"], [["A", "B"]])
        self.assertEqual(summary["failed_ids"], ["C"])
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
