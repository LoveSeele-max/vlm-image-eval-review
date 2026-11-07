"""Apply the C04 whole-image contract without modifying original files."""

import hashlib
import io

from PIL import Image, ImageOps, __version__ as pillow_version


def prepare_image(path, role, policy, expected_sha256=None):
    if role not in policy["image_order"]:
        raise ValueError("未知图片角色。")
    if (policy.get("crop") is not False or policy.get("upscale") is not False
            or policy.get("color_mode") != "RGB" or policy.get("format") != "PNG"
            or policy.get("resample") != "LANCZOS"
            or policy.get("apply_exif_transpose") is not True):
        raise ValueError("不支持的图片预处理规则。")
    raw = path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError("原图 SHA-256 与固定记录不符。")
    with Image.open(io.BytesIO(raw)) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("只接受单帧图。")
        original_size = list(source.size)
        image = ImageOps.exif_transpose(source).convert("RGB")
        limit = policy["candidate_max_long_edge" if role == "CANDIDATE"
                       else "reference_max_long_edge"]
        if type(limit) is not int or limit <= 0:
            raise ValueError("图片长边限制非法。")
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        receipt = {
            "role": role, "source_sha256": source_sha, "source_size": original_size,
            "output_size": list(image.size), "output_sha256": hashlib.sha256(png).hexdigest(),
            "output_bytes": len(png), "mode": image.mode, "format": "PNG",
            "pillow_version": pillow_version, "policy_id": policy["policy_id"],
        }
    return png, receipt
