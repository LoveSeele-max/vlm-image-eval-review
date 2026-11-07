import copy
import json
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capability_probe import COLORS, SHAPES, create_challenge, make_png, validate_answer


class ProbeTests(unittest.TestCase):
    def test_png_integrity_pixels_and_separate_shape_counts(self):
        for shape in SHAPES:
            for count in (1, 2, 3):
                with self.subTest(shape=shape, count=count):
                    png = make_png("blue", shape, count)
                    self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                    offset, compressed = 8, b""
                    while offset < len(png):
                        length = struct.unpack(">I", png[offset:offset + 4])[0]
                        kind = png[offset + 4:offset + 8]
                        data = png[offset + 8:offset + 8 + length]
                        crc = struct.unpack(">I", png[offset + 8 + length:offset + 12 + length])[0]
                        self.assertEqual(crc, zlib.crc32(kind + data) & 0xffffffff)
                        if kind == b"IHDR":
                            self.assertEqual(struct.unpack(">2I5B", data), (256, 256, 8, 2, 0, 0, 0))
                        elif kind == b"IDAT":
                            compressed += data
                        offset += length + 12
                    rows = zlib.decompress(compressed)
                    self.assertEqual(len(rows), 256 * 769)
                    foreground = set()
                    for y in range(256):
                        self.assertEqual(rows[y * 769], 0)
                        for x in range(256):
                            rgb = tuple(rows[y * 769 + 1 + x * 3:y * 769 + 4 + x * 3])
                            self.assertIn(rgb, ((255, 255, 255), COLORS["blue"]))
                            if rgb == COLORS["blue"]:
                                self.assertTrue(0 < x < 255 and 0 < y < 255)
                                foreground.add((x, y))
                    components = []
                    while foreground:
                        stack = [foreground.pop()]
                        component = set(stack)
                        while stack:
                            x, y = stack.pop()
                            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                                if neighbor in foreground:
                                    foreground.remove(neighbor)
                                    component.add(neighbor)
                                    stack.append(neighbor)
                        components.append(component)
                    self.assertEqual(len(components), count)
                    for component in components:
                        ys = [y for _, y in component]
                        widths = [sum(y == row for _, y in component)
                                  for row in range(min(ys), max(ys) + 1)]
                        if shape == "square":
                            self.assertEqual(set(widths), {71})
                        elif shape == "circle":
                            self.assertEqual(widths, list(reversed(widths)))
                            self.assertLess(widths[0], widths[len(widths) // 2])
                        else:
                            self.assertEqual(widths, sorted(widths))
                            self.assertLess(widths[0], widths[-1])

    def test_challenge_has_distinct_images_and_local_metadata(self):
        challenge = create_challenge()
        self.assertEqual([image["image_id"] for image in challenge["images"]], [1, 2, 3])
        self.assertEqual(len({image["color"] for image in challenge["images"]}), 3)
        self.assertEqual({image["shape"] for image in challenge["images"]}, set(SHAPES))
        self.assertEqual(len({image["sha256"] for image in challenge["images"]}), 3)

    def test_json_contract_rejects_missing_extra_enums_wrong_types_and_nonstandard_json(self):
        valid = {"images": [
            {"image_id": 1, "color": "red", "shape": "circle", "count": 1},
            {"image_id": 2, "color": "green", "shape": "square", "count": 2},
            {"image_id": 3, "color": "blue", "shape": "triangle", "count": 3},
        ]}
        self.assertEqual(validate_answer(json.dumps(valid)), (valid["images"], None))
        invalid = ["[]", "null", "```json\n{}\n```", '{"images":[],"images":[]}',
                   '{"images":NaN}', '{"images":Infinity}']
        for field, bad in (("image_id", True), ("image_id", 1.0), ("image_id", 4),
                           ("count", False), ("count", "1"), ("count", 0),
                           ("color", "yellow"), ("color", []), ("shape", "star")):
            value = copy.deepcopy(valid)
            value["images"][0][field] = bad
            invalid.append(json.dumps(value))
        for mutation in ("missing", "extra", "short", "long", "root_extra"):
            value = copy.deepcopy(valid)
            if mutation == "missing":
                del value["images"][0]["color"]
            elif mutation == "extra":
                value["images"][0]["extra"] = "hidden"
            elif mutation == "short":
                value["images"].pop()
            elif mutation == "long":
                value["images"].append(value["images"][0])
            else:
                value["extra"] = 1
            invalid.append(json.dumps(value))
        for text in invalid:
            with self.subTest(text=text):
                images, error = validate_answer(text)
                self.assertIsNone(images)
                self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
