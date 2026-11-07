"""Synthetic three-image smoke test, without external packages or dataset images."""

import base64
import hashlib
import json
import secrets
import struct
import zlib

from api_connection import send_probe

TEST_TYPE = "multi_image_json"
COLORS = {
    "red": (220, 35, 35), "green": (25, 155, 50), "blue": (35, 75, 220),
    "orange": (245, 140, 15), "purple": (145, 45, 190), "cyan": (10, 185, 200),
}
SHAPES = ("circle", "square", "triangle")
DESCRIPTOR_KEYS = {"image_id", "color", "shape", "count"}
SCHEMA = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "integer", "enum": [1, 2, 3]},
                    "color": {"type": "string", "enum": list(COLORS)},
                    "shape": {"type": "string", "enum": list(SHAPES)},
                    "count": {"type": "integer", "enum": [1, 2, 3]},
                },
                "required": ["image_id", "color", "shape", "count"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["images"],
    "additionalProperties": False,
}
PROMPT = (
    "Inspect all three attached images in their attachment order. "
    "Each image has a white background and one to three identical solid shapes. "
    "Return only the JSON object required by the schema, with exactly three images entries "
    "in attachment order. Use image_id 1, 2, 3 respectively. "
    "For each image identify the foreground color, shape and number of shapes. "
    "Allowed colors: red, green, blue, orange, purple, cyan. "
    "Allowed shapes: circle, square, triangle. Do not include explanations or Markdown."
)


def make_png(color, shape, count):
    """Render separated, large shapes into a 256x256 RGB PNG."""
    size, radius = 256, 35
    pixels = bytearray(b"\xff" * (size * size * 3))
    centers = {
        1: [(128, 128)], 2: [(64, 128), (192, 128)],
        3: [(64, 64), (192, 64), (128, 184)],
    }[count]
    rgb = bytes(COLORS[color])
    for cx, cy in centers:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                inside = (
                    shape == "square"
                    or (shape == "circle" and dx * dx + dy * dy <= radius * radius)
                    or (shape == "triangle" and 2 * abs(dx) <= dy + radius)
                )
                if inside:
                    offset = ((cy + dy) * size + cx + dx) * 3
                    pixels[offset:offset + 3] = rgb
    rows = b"".join(b"\x00" + pixels[y * size * 3:(y + 1) * size * 3] for y in range(size))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">2I5B", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def create_challenge():
    rng = secrets.SystemRandom()
    colors = rng.sample(list(COLORS), 3)
    shapes = rng.sample(list(SHAPES), 3)
    images = []
    for index, (color, shape) in enumerate(zip(colors, shapes), 1):
        count = rng.randint(1, 3)
        png = make_png(color, shape, count)
        images.append({
            "image_id": index, "color": color, "shape": shape, "count": count,
            "png_base64": base64.b64encode(png).decode("ascii"),
            "sha256": hashlib.sha256(png).hexdigest(),
        })
    return {"fixture_version": "colored_shapes_v1", "images": images}


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("返回 JSON 含重复字段。")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("返回 JSON 含 NaN 或 Infinity。")


def validate_answer(text):
    try:
        value = json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return None, "不是严格 JSON，或包含重复字段、非法数值。"
    if not isinstance(value, dict) or set(value) != {"images"}:
        return None, "顶层必须只有 images 字段。"
    images = value["images"]
    if not isinstance(images, list) or len(images) != 3:
        return None, "images 必须包含恰好 3 项。"
    for index, item in enumerate(images, 1):
        if not isinstance(item, dict) or set(item) != DESCRIPTOR_KEYS:
            return None, f"第 {index} 项存在缺失或额外字段。"
        for key in ("image_id", "count"):
            if type(item[key]) is not int or item[key] not in (1, 2, 3):
                return None, f"第 {index} 项的 {key} 必须是整数 1、2 或 3。"
        if not isinstance(item["color"], str) or item["color"] not in COLORS:
            return None, f"第 {index} 项的 color 不在允许范围。"
        if not isinstance(item["shape"], str) or item["shape"] not in SHAPES:
            return None, f"第 {index} 项的 shape 不在允许范围。"
    return images, None


def test_capabilities(base_url, model, challenge, timeout=120):
    content = [{"type": "input_text", "text": PROMPT}]
    content.extend({
        "type": "input_image",
        "image_url": "data:image/png;base64," + image["png_base64"],
        "detail": "low",
    } for image in challenge["images"])
    receipt = send_probe(
        base_url, model, [{"role": "user", "content": content}], timeout=timeout,
        test_type=TEST_TYPE, text_limit=8000,
        text_format={"type": "json_schema", "name": "vision_capability_probe",
                     "strict": True, "schema": SCHEMA},
    )
    expected = [{key: image[key] for key in ("image_id", "color", "shape", "count")}
                for image in challenge["images"]]
    receipt.update({
        "request_completed": True,
        "fixture_version": challenge["fixture_version"],
        "image_sha256": [image["sha256"] for image in challenge["images"]],
        "expected": expected,
        "schema_valid": False, "visual_match": None, "success": False,
        "schema_enforcement": "not_proven_by_smoke_test",
    })
    actual, error = validate_answer(receipt["text"])
    if receipt["text_truncated"]:
        actual, error = None, "输出过长，未作为有效结构化结果接收。"
    receipt["schema_error"] = error
    if actual is not None:
        receipt["schema_valid"] = True
        receipt["actual"] = actual
        receipt["checks"] = [
            {"position": index, "expected": wanted, "actual": got, "match": wanted == got}
            for index, (wanted, got) in enumerate(zip(expected, actual), 1)
        ]
        receipt["visual_match"] = all(item["match"] for item in receipt["checks"])
        receipt["success"] = receipt["visual_match"]
    return receipt
