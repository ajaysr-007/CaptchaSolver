import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytesseract
from PIL import Image

_cv2 = importlib.import_module("cv2")  # type: Any

# Ensure Tesseract path from environment is picked up if supplied.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def load_and_normalise(path_or_array) -> np.ndarray:
    """
    Load any CAPTCHA image (file path OR numpy array) and return an 8-bit
    single-channel grayscale image, correcting for inverted palettes.
    """
    if isinstance(path_or_array, (str, Path)):
        img = _cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Cannot open: {path_or_array}")
    else:
        img = path_or_array.copy()

    if img.ndim == 3:
        lab = _cv2.cvtColor(img, _cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]
    else:
        gray = img

    if gray.mean() < 128:
        gray = _cv2.bitwise_not(gray)

    return gray


def binarise(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert grayscale → binary.
    """
    if method == "global":
        _, bw = _cv2.threshold(gray, int(gray.mean() * 0.85), 255, _cv2.THRESH_BINARY)
    elif method == "otsu":
        blurred = _cv2.GaussianBlur(gray, (5, 5), 0)
        _, bw = _cv2.threshold(blurred, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
    else:
        bw = _cv2.adaptiveThreshold(
            gray,
            255,
            _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            _cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=4,
        )

    if np.sum(bw == 255) > np.sum(bw == 0):
        bw = _cv2.bitwise_not(bw)

    return _cv2.bitwise_not(bw)


def remove_noise_blobs(bw: np.ndarray, min_area: int = 50, min_height: int = 10) -> np.ndarray:
    """
    Remove tiny speckles and isolated blobs that are not characters.
    """
    inv = _cv2.bitwise_not(bw)
    nb, labels, stats, _ = _cv2.connectedComponentsWithStats(inv, connectivity=8)
    cleaned = np.zeros_like(inv)

    for i in range(1, nb):
        area = stats[i, _cv2.CC_STAT_AREA]
        height = stats[i, _cv2.CC_STAT_HEIGHT]
        if area >= min_area and height >= min_height:
            cleaned[labels == i] = 255

    return _cv2.bitwise_not(cleaned)


def remove_lines(
    bw: np.ndarray,
    hough_threshold: int = 40,
    min_line_length: int = 60,
    max_line_gap: int = 15,
    max_angle_deg: float = 30.0,
    thickness: int = 5,
) -> np.ndarray:
    """
    Detect and remove near-horizontal distortion lines by using a Hough transform.
    """
    inv = _cv2.bitwise_not(bw)
    edges = _cv2.Canny(inv, 40, 120)
    lines = _cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    line_mask = np.zeros_like(inv)
    if lines is not None:
        for segment in lines:
            x1, y1, x2, y2 = segment[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= max_angle_deg:
                _cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness)

    kernel = np.ones((3, 3), np.uint8)
    line_mask = _cv2.dilate(line_mask, kernel, iterations=1)
    result_inv = _cv2.subtract(inv, line_mask)
    return _cv2.bitwise_not(result_inv)


def morphological_cleanup(bw: np.ndarray) -> np.ndarray:
    """
    Perform morphological closing and opening to repair character strokes and remove residual noise.
    """
    inv = _cv2.bitwise_not(bw)
    close_kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    inv = _cv2.morphologyEx(inv, _cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((2, 2), np.uint8)
    inv = _cv2.morphologyEx(inv, _cv2.MORPH_OPEN, open_kernel)
    return _cv2.bitwise_not(inv)


def upscale(bw: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Upscale the binary image to improve OCR accuracy on small characters.
    """
    h, w = bw.shape
    big = _cv2.resize(bw, (w * scale, h * scale), interpolation=_cv2.INTER_LANCZOS4)
    _, big = _cv2.threshold(big, 127, 255, _cv2.THRESH_BINARY)
    return big


def segment_characters(
    bw_upscaled: np.ndarray,
    min_area: int = 400,
    min_height: int = 30,
    min_width: int = 10,
    merge_overlap_px: int = 8,
) -> list[tuple]:
    """
    Return sorted bounding boxes for candidate character regions.
    """
    inv = _cv2.bitwise_not(bw_upscaled)
    nb, _, stats, _ = _cv2.connectedComponentsWithStats(inv, connectivity=8)
    boxes = []

    for i in range(1, nb):
        x, y = stats[i, _cv2.CC_STAT_LEFT], stats[i, _cv2.CC_STAT_TOP]
        w, h = stats[i, _cv2.CC_STAT_WIDTH], stats[i, _cv2.CC_STAT_HEIGHT]
        area = stats[i, _cv2.CC_STAT_AREA]
        if area >= min_area and h >= min_height and w >= min_width:
            boxes.append([x, y, w, h])

    boxes.sort(key=lambda b: b[0])
    merged = []

    for box in boxes:
        if not merged:
            merged.append(box)
        else:
            prev = merged[-1]
            prev_end = prev[0] + prev[2]
            cur_start = box[0]
            if cur_start - prev_end <= merge_overlap_px:
                new_x = min(prev[0], box[0])
                new_y = min(prev[1], box[1])
                new_x2 = max(prev[0] + prev[2], box[0] + box[2])
                new_y2 = max(prev[1] + prev[3], box[1] + box[3])
                merged[-1] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                merged.append(box)

    return [tuple(b) for b in merged]


WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def ocr_full_image(bw: np.ndarray) -> str:
    """
    Run Tesseract on the full image; choose the longest plausible result.
    """
    pil = Image.fromarray(bw)
    best = ""
    for psm in (7, 8, 6, 13):
        cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        text = pytesseract.image_to_string(pil, config=cfg).strip()
        text = "".join(c for c in text if c in WHITELIST)
        if len(text) > len(best):
            best = text
    return best


def ocr_per_character(bw: np.ndarray, boxes: list[tuple], pad: int = 6) -> str:
    """
    Crop each detected character region and run single-character OCR.
    """
    if not boxes:
        return ocr_full_image(bw)

    chars = []
    cfg = f"--psm 10 -c tessedit_char_whitelist={WHITELIST}"

    for (x, y, w, h) in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(bw.shape[1], x + w + pad)
        y1 = min(bw.shape[0], y + h + pad)
        roi = bw[y0:y1, x0:x1]
        roi_big = _cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=_cv2.INTER_LANCZOS4)
        _, roi_big = _cv2.threshold(roi_big, 127, 255, _cv2.THRESH_BINARY)
        pil_roi = Image.fromarray(roi_big)
        c = pytesseract.image_to_string(pil_roi, config=cfg).strip()
        c = "".join(ch for ch in c if ch in WHITELIST)
        chars.append(c if c else "?")

    return "".join(chars)


class CaptchaSolver:
    def __init__(
        self,
        threshold_method: str = "adaptive",
        noise_min_area: int = 50,
        noise_min_height: int = 10,
        hough_threshold: int = 40,
        min_line_length: int = 60,
        max_line_angle: float = 30.0,
        line_thickness: int = 5,
        upscale_factor: int = 4,
        char_min_area: int = 400,
        char_min_height: int = 30,
        ocr_mode: str = "per_char",
        line_removal_method: str = "morphology",
        morph_erode_kernel: tuple = (3, 2),
        morph_dilate_kernel: tuple = (3, 2),
        post_noise_min_area: int = 30,
    ):
        self.threshold_method = threshold_method
        self.noise_min_area = noise_min_area
        self.noise_min_height = noise_min_height
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_angle = max_line_angle
        self.line_thickness = line_thickness
        self.upscale_factor = upscale_factor
        self.char_min_area = char_min_area
        self.char_min_height = char_min_height
        self.ocr_mode = ocr_mode
        self.line_removal_method = line_removal_method
        self.morph_erode_kernel = morph_erode_kernel
        self.morph_dilate_kernel = morph_dilate_kernel
        self.post_noise_min_area = post_noise_min_area

    def step_load(self, source) -> np.ndarray:
        return load_and_normalise(source)

    def step_binarise(self, gray: np.ndarray) -> np.ndarray:
        return binarise(gray, method=self.threshold_method)

    def step_remove_noise(self, bw: np.ndarray) -> np.ndarray:
        return remove_noise_blobs(bw, self.noise_min_area, self.noise_min_height)

    def step_remove_lines(self, bw: np.ndarray) -> np.ndarray:
        return remove_lines(
            bw,
            hough_threshold=self.hough_threshold,
            min_line_length=self.min_line_length,
            max_angle_deg=self.max_line_angle,
            thickness=self.line_thickness,
        )

    def step_morph(self, bw: np.ndarray) -> np.ndarray:
        return morphological_cleanup(bw)

    def step_morph_line_removal(self, bw: np.ndarray) -> np.ndarray:
        inv = _cv2.bitwise_not(bw)
        kernel_erode = _cv2.getStructuringElement(_cv2.MORPH_RECT, self.morph_erode_kernel)
        eroded = _cv2.erode(inv, kernel_erode, iterations=1)
        kernel_dilate = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, self.morph_dilate_kernel)
        dilated = _cv2.dilate(eroded, kernel_dilate, iterations=1)
        bw_post = _cv2.bitwise_not(dilated)
        return remove_noise_blobs(bw_post, min_area=self.post_noise_min_area, min_height=5)

    def step_upscale(self, bw: np.ndarray) -> np.ndarray:
        return upscale(bw, self.upscale_factor)

    def step_segment(self, bw_up: np.ndarray) -> list[tuple]:
        return segment_characters(
            bw_up,
            min_area=self.char_min_area,
            min_height=self.char_min_height,
        )

    def solve(self, source, verbose: bool = False) -> str:
        gray = self.step_load(source)
        bw = self.step_binarise(gray)
        bw = self.step_remove_noise(bw)
        
        if self.line_removal_method == "hough":
            bw = self.step_remove_lines(bw)
            bw = self.step_morph(bw)
        elif self.line_removal_method == "morphology":
            bw = self.step_morph_line_removal(bw)
        else:
            bw = self.step_morph(bw)
            
        bw_up = self.step_upscale(bw)
        boxes = self.step_segment(bw_up)

        if verbose:
            print(f"[CaptchaSolver] {len(boxes)} character region(s) detected.")

        if self.ocr_mode == "full":
            return ocr_full_image(bw_up)

        result_per = ocr_per_character(bw_up, boxes)
        result_full = ocr_full_image(bw_up)
        if len(boxes) < 3:
            return result_full
        return result_per if len(result_per) >= len(result_full) else result_full

    def debug_steps(self, source) -> dict:
        gray = self.step_load(source)
        bw = self.step_binarise(gray)
        bw_nb = self.step_remove_noise(bw)
        
        if self.line_removal_method == "hough":
            bw_nl = self.step_remove_lines(bw_nb)
            bw_m = self.step_morph(bw_nl)
        elif self.line_removal_method == "morphology":
            bw_m = self.step_morph_line_removal(bw_nb)
            bw_nl = bw_m
        else:
            bw_m = self.step_morph(bw_nb)
            bw_nl = bw_m
            
        bw_up = self.step_upscale(bw_m)
        boxes = self.step_segment(bw_up)
        text = ocr_per_character(bw_up, boxes) if self.ocr_mode == "per_char" else ocr_full_image(bw_up)
        return {
            "gray": gray,
            "binarised": bw,
            "noise_removed": bw_nb,
            "lines_removed": bw_nl,
            "morph_cleaned": bw_m,
            "upscaled": bw_up,
            "char_boxes": boxes,
            "result": text,
        }
