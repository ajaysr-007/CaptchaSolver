"""
Captcha Solver Package.
Exposes the CaptchaSolver class and helper pre-processing functions.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

from .solver import (
    CaptchaSolver,
    load_and_normalise,
    load_color,
    crop_yellow_line,
    enhance_contrast,
    median_denoise,
    binarise,
    remove_noise_blobs,
    remove_lines,
    remove_lines_by_components,
    morphological_cleanup,
    upscale,
    segment_characters,
    ocr_full_image,
    ocr_per_character,
    ocr_multi_psm,
    vote_on_results,
)

__all__ = [
    "CaptchaSolver",
    "load_and_normalise",
    "load_color",
    "crop_yellow_line",
    "enhance_contrast",
    "median_denoise",
    "binarise",
    "remove_noise_blobs",
    "remove_lines",
    "remove_lines_by_components",
    "morphological_cleanup",
    "upscale",
    "segment_characters",
    "ocr_full_image",
    "ocr_per_character",
    "ocr_multi_psm",
    "vote_on_results",
]
