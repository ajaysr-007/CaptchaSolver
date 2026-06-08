import base64
import importlib
import json
import os
import tempfile

import azure.functions as func
import numpy as np
from captcha_solver import CaptchaSolver

_cv2 = importlib.import_module("cv2")  # type: ignore


def main(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_json(silent=True) or {}
    image_path = body.get("image_path")
    image_base64 = body.get("image_base64")

    if not image_path and not image_base64:
        return func.HttpResponse(
            json.dumps({"error": "Provide image_path or image_base64 in JSON body."}),
            status_code=400,
            mimetype="application/json",
        )

    if image_base64:
        try:
            image_data = base64.b64decode(image_base64)
            arr = np.frombuffer(image_data, dtype=np.uint8)
            image = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image_base64")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                _cv2.imwrite(tmp_path, image)
            image_path = tmp_path
        except (ValueError, base64.binascii.Error) as exc:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid image_base64: {exc}"}),
                status_code=400,
                mimetype="application/json",
            )

    try:
        solver = CaptchaSolver()
        text = solver.solve(image_path)
    except FileNotFoundError as exc:
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=404,
            mimetype="application/json",
        )
    finally:
        if image_base64 and image_path and os.path.exists(image_path):
            os.remove(image_path)

    return func.HttpResponse(
        json.dumps({"text": text}),
        status_code=200,
        mimetype="application/json",
    )
