import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

from .solver import CaptchaSolver, load_and_normalise, binarise, remove_noise_blobs, remove_lines, morphological_cleanup, upscale, segment_characters, ocr_full_image, ocr_per_character

__all__ = [
    "CaptchaSolver",
    "load_and_normalise",
    "binarise",
    "remove_noise_blobs",
    "remove_lines",
    "morphological_cleanup",
    "upscale",
    "segment_characters",
    "ocr_full_image",
    "ocr_per_character",
]
