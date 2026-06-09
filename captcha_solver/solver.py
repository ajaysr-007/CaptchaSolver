import importlib
import os
from collections import Counter
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


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_color(path_or_array) -> np.ndarray:
    """
    Load a CAPTCHA image and return the BGR colour image.
    Accepts a file path or a numpy array.
    """
    if isinstance(path_or_array, (str, Path)):
        img = _cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Cannot open: {path_or_array}")
        return img
    return path_or_array.copy()


def crop_yellow_line(bgr: np.ndarray, threshold: int = 180) -> np.ndarray:
    """
    Detect and crop out yellow/gold horizontal lines (common in Type C captchas).
    Finds rows where a majority of pixels are yellowish and replaces them with white.
    """
    hsv = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2HSV)
    # Yellow in HSV: hue 15-35, high saturation, high value
    lower_yellow = np.array([15, 50, 150])
    upper_yellow = np.array([35, 255, 255])
    mask = _cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find rows where more than 30% of pixels are yellow
    row_yellow_pct = np.sum(mask > 0, axis=1) / mask.shape[1]
    yellow_rows = row_yellow_pct > 0.3

    if np.any(yellow_rows):
        result = bgr.copy()
        result[yellow_rows] = [255, 255, 255]
        return result
    return bgr


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

    # Crop yellow line before grayscale conversion
    if img.ndim == 3:
        img = crop_yellow_line(img)
        lab = _cv2.cvtColor(img, _cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]
    else:
        gray = img

    if gray.mean() < 128:
        gray = _cv2.bitwise_not(gray)

    return gray


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------

def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    boost text contrast against noisy backgrounds.
    """
    clahe = _cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def median_denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply median blur to remove salt-and-pepper dot noise before binarisation.
    """
    return _cv2.medianBlur(gray, ksize)


def binarise(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert grayscale to binary.
    """
    if method == "global":
        _, bw = _cv2.threshold(gray, int(gray.mean() * 0.85), 255, _cv2.THRESH_BINARY)
    elif method == "otsu":
        blurred = _cv2.GaussianBlur(gray, (5, 5), 0)
        _, bw = _cv2.threshold(blurred, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
    elif method == "sauvola":
        # Sauvola-like local thresholding using adaptive with tuned params
        bw = _cv2.adaptiveThreshold(
            gray,
            255,
            _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            _cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=8,
        )
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


def remove_lines_by_components(bw: np.ndarray, min_aspect_ratio: float = 5.0, min_width: int = 40) -> np.ndarray:
    """
    Remove long, thin connected components that are likely distortion lines
    rather than characters.  Uses aspect ratio (width/height or height/width)
    to distinguish lines from letter strokes.
    """
    inv = _cv2.bitwise_not(bw)
    nb, labels, stats, _ = _cv2.connectedComponentsWithStats(inv, connectivity=8)
    cleaned = inv.copy()

    for i in range(1, nb):
        w = stats[i, _cv2.CC_STAT_WIDTH]
        h = stats[i, _cv2.CC_STAT_HEIGHT]
        area = stats[i, _cv2.CC_STAT_AREA]
        if h == 0 or w == 0:
            continue
        aspect = max(w / h, h / w)
        # Long thin component with wide extent - line
        if aspect >= min_aspect_ratio and max(w, h) >= min_width:
            # Check density - lines are sparse, characters are dense
            density = area / (w * h)
            if density < 0.4:
                cleaned[labels == i] = 0

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


# ---------------------------------------------------------------------------
# OCR functions
# ---------------------------------------------------------------------------

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
WHITELIST_MIXED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _pad_image(bw: np.ndarray, pad: int = 20) -> np.ndarray:
    """Add white padding around the image. Tesseract performs better with border space."""
    return _cv2.copyMakeBorder(bw, pad, pad, pad, pad, _cv2.BORDER_CONSTANT, value=255)


def _sharpen(bw: np.ndarray) -> np.ndarray:
    """Sharpen edges to help Tesseract distinguish similar characters."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = _cv2.filter2D(bw, -1, kernel)
    _, sharpened = _cv2.threshold(sharpened, 127, 255, _cv2.THRESH_BINARY)
    return sharpened


def ocr_full_image(bw: np.ndarray) -> str:
    """
    Run Tesseract on the full image; choose the longest plausible result.
    Always returns uppercase since CAPTCHAs are case-insensitive.
    """
    padded = _pad_image(bw)
    pil = Image.fromarray(padded)
    best = ""
    for psm in (7, 8, 6, 13):
        cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        text = pytesseract.image_to_string(pil, config=cfg).strip()
        text = "".join(c for c in text if c.isalnum()).upper()
        if len(text) > len(best):
            best = text
    return best


def ocr_per_character(bw: np.ndarray, boxes: list[tuple], pad: int = 6) -> str:
    """
    Crop each detected character region and run single-character OCR.
    Takes only the first valid character per box and uppercases output
    to prevent double-reads (e.g. 'YY' -> 'Y') and case errors (e.g. 'y' -> 'Y').
    """
    if not boxes:
        return ocr_full_image(bw)

    chars = []

    for (x, y, w, h) in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(bw.shape[1], x + w + pad)
        y1 = min(bw.shape[0], y + h + pad)
        roi = bw[y0:y1, x0:x1]
        roi_big = _cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2), interpolation=_cv2.INTER_LANCZOS4)
        _, roi_big = _cv2.threshold(roi_big, 127, 255, _cv2.THRESH_BINARY)
        roi_padded = _pad_image(roi_big, pad=10)
        pil_roi = Image.fromarray(roi_padded)
        best_char = ""
        for psm in (10, 8, 7):
            cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            c = pytesseract.image_to_string(pil_roi, config=cfg).strip()
            c = "".join(ch for ch in c if ch.isalnum()).upper()
            if c:
                best_char = c[0]
                break
        chars.append(best_char if best_char else "?")

    return "".join(chars)


def ocr_multi_psm(bw: np.ndarray) -> list[str]:
    """
    Run Tesseract with multiple PSM modes and return all results.
    Uses both padded and sharpened variants for diversity.
    """
    padded = _pad_image(bw)
    sharpened = _pad_image(_sharpen(bw))
    results = []

    for img in (padded, sharpened):
        pil = Image.fromarray(img)
        for psm in (7, 8, 6):
            cfg = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            text = pytesseract.image_to_string(pil, config=cfg).strip()
            text = "".join(c for c in text if c.isalnum()).upper()
            if text:
                results.append(text)

    return results


# ---------------------------------------------------------------------------
# Voting / consensus logic
# ---------------------------------------------------------------------------

EXPECTED_LENGTH = 5


def _trim_to_length(candidates: list[str], expected_len: int) -> list[str]:
    """
    For candidates that are slightly longer than expected_len,
    try trimming from the start or end to produce expected_len candidates.
    This helps when distortion lines cause Tesseract to read extra chars.
    """
    trimmed = []
    for c in candidates:
        if len(c) == expected_len:
            trimmed.append(c)
        elif len(c) == expected_len + 1:
            # Try removing first or last char
            trimmed.append(c[1:])
            trimmed.append(c[:-1])
        elif len(c) == expected_len + 2:
            # Try removing from both ends
            trimmed.append(c[1:-1])
    return trimmed


def vote_on_results(candidates: list[str], expected_len: int = EXPECTED_LENGTH) -> str:
    """
    Given a list of OCR candidate strings, use character-position voting
    to produce the best consensus result.

    1. Prefer candidates with exactly the expected length.
    2. Try trimming overlong candidates to the expected length.
    3. For each character position, pick the most common character across candidates.
    4. If no candidate has the expected length, return the most common full string.
    """
    if not candidates:
        return ""

    # Filter to candidates with exactly the expected length
    exact = [c for c in candidates if len(c) == expected_len]

    # If not enough exact matches, try trimming overlong candidates
    if len(exact) < 3:
        trimmed = _trim_to_length(candidates, expected_len)
        exact = exact + trimmed

    if exact:
        # Position-wise voting
        result_chars = []
        for pos in range(expected_len):
            chars_at_pos = [c[pos] for c in exact]
            counter = Counter(chars_at_pos)
            result_chars.append(counter.most_common(1)[0][0])
        return "".join(result_chars)

    # Fallback: most common full result, or longest
    counter = Counter(candidates)
    most_common = counter.most_common(1)[0][0]
    return most_common


# ---------------------------------------------------------------------------
# Pipeline definitions for multi-pipeline voting
# ---------------------------------------------------------------------------

def _run_pipeline_adaptive(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pipeline 1: CLAHE + median + adaptive threshold + component line removal + morph + upscale"""
    enhanced = enhance_contrast(gray)
    denoised = median_denoise(enhanced, ksize=3)
    bw = binarise(denoised, method="adaptive")
    bw = remove_noise_blobs(bw, min_area=50, min_height=10)
    bw = remove_lines_by_components(bw)
    bw = morphological_cleanup(bw)
    return upscale(bw, scale)


def _run_pipeline_otsu(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pipeline 2: CLAHE + Otsu threshold + noise removal + component line removal + morph + upscale"""
    enhanced = enhance_contrast(gray, clip_limit=3.0)
    denoised = median_denoise(enhanced, ksize=3)
    bw = binarise(denoised, method="otsu")
    bw = remove_noise_blobs(bw, min_area=40, min_height=8)
    bw = remove_lines_by_components(bw)
    bw = morphological_cleanup(bw)
    return upscale(bw, scale)


def _run_pipeline_sauvola(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pipeline 3: median + sauvola threshold + morph cleanup + upscale"""
    denoised = median_denoise(gray, ksize=3)
    bw = binarise(denoised, method="sauvola")
    bw = remove_noise_blobs(bw, min_area=60, min_height=10)
    bw = morphological_cleanup(bw)
    return upscale(bw, scale)


def _run_pipeline_global(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pipeline 4: CLAHE + global threshold + noise removal + morph + upscale"""
    enhanced = enhance_contrast(gray, clip_limit=2.5)
    bw = binarise(enhanced, method="global")
    bw = remove_noise_blobs(bw, min_area=30, min_height=8)
    bw = morphological_cleanup(bw)
    return upscale(bw, scale)


def _run_pipeline_bilateral(gray: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pipeline 5: Bilateral filter (edge-preserving) + Otsu + component line removal + upscale"""
    # Bilateral filter preserves edges while smoothing noise
    filtered = _cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    enhanced = enhance_contrast(filtered, clip_limit=2.0)
    bw = binarise(enhanced, method="otsu")
    bw = remove_noise_blobs(bw, min_area=40, min_height=8)
    bw = remove_lines_by_components(bw)
    bw = morphological_cleanup(bw)
    return upscale(bw, scale)


PIPELINES = [
    ("adaptive", _run_pipeline_adaptive),
    ("otsu", _run_pipeline_otsu),
    ("sauvola", _run_pipeline_sauvola),
    ("global", _run_pipeline_global),
    ("bilateral", _run_pipeline_bilateral),
]


# ---------------------------------------------------------------------------
# CaptchaSolver class
# ---------------------------------------------------------------------------

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
        char_min_area: int = 300,
        char_min_height: int = 25,
        ocr_mode: str = "per_char",
        line_removal_method: str = "morphology",
        morph_erode_kernel: tuple = (3, 3),
        morph_dilate_kernel: tuple = (3, 3),
        post_noise_min_area: int = 30,
        use_voting: bool = True,
        expected_length: int = EXPECTED_LENGTH,
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
        self.use_voting = use_voting
        self.expected_length = expected_length

    # --- Single-pipeline step helpers (kept for backward compatibility) ---

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

    # --- Multi-pipeline voting solver ---

    def solve_voting(self, source, verbose: bool = False) -> str:
        """
        Run multiple preprocessing pipelines and OCR modes, then use
        character-position voting to produce the best consensus result.
        """
        gray = self.step_load(source)
        all_candidates = []

        for name, pipeline_fn in PIPELINES:
            try:
                bw_up = pipeline_fn(gray, scale=self.upscale_factor)

                # Full-image OCR with multiple PSM modes
                full_results = ocr_multi_psm(bw_up)
                all_candidates.extend(full_results)

                # Per-character OCR
                boxes = self.step_segment(bw_up)
                if boxes:
                    per_char_result = ocr_per_character(bw_up, boxes)
                    if per_char_result and "?" not in per_char_result:
                        all_candidates.append(per_char_result)

                if verbose:
                    print(f"  [{name}] full={full_results}, boxes={len(boxes)}")

            except Exception as e:
                if verbose:
                    print(f"  [{name}] FAILED: {e}")
                continue

        if verbose:
            print(f"  All candidates ({len(all_candidates)}): {all_candidates}")

        result = vote_on_results(all_candidates, self.expected_length)

        if verbose:
            print(f"  Voted result: {result}")

        return result

    # --- Legacy single-pipeline solver ---

    def solve_single(self, source, verbose: bool = False) -> str:
        """
        Original single-pipeline solver (kept for backward compatibility).
        """
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
        # Need at least 4 char boxes for per-char mode to be trustworthy
        if len(boxes) < 4:
            return result_full
        # Prefer per-char result: it handles individual chars better.
        # Fall back to full-image if per-char produced fewer characters.
        return result_per if len(result_per) >= len(result_full) else result_full

    # --- Main solve entry point ---

    def solve(self, source, verbose: bool = False) -> str:
        """
        Solve a CAPTCHA image.  Uses multi-pipeline voting by default
        (use_voting=True) for best accuracy, or the legacy single pipeline.
        """
        if self.use_voting:
            return self.solve_voting(source, verbose=verbose)
        return self.solve_single(source, verbose=verbose)

    def debug_steps(self, source) -> dict:
        """
        Run all pipelines and return intermediate images plus per-pipeline results
        for debugging and visual inspection.
        """
        gray = self.step_load(source)

        pipeline_results = {}
        all_candidates = []

        for name, pipeline_fn in PIPELINES:
            try:
                bw_up = pipeline_fn(gray, scale=self.upscale_factor)
                full_results = ocr_multi_psm(bw_up)
                all_candidates.extend(full_results)

                boxes = self.step_segment(bw_up)
                per_char = ocr_per_character(bw_up, boxes) if boxes else ""
                if per_char and "?" not in per_char:
                    all_candidates.append(per_char)

                pipeline_results[name] = {
                    "upscaled": bw_up,
                    "boxes": boxes,
                    "full_results": full_results,
                    "per_char": per_char,
                }
            except Exception as e:
                pipeline_results[name] = {"error": str(e)}

        voted = vote_on_results(all_candidates, self.expected_length)

        # Also run the legacy single pipeline for comparison
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
        legacy_text = ocr_per_character(bw_up, boxes) if self.ocr_mode == "per_char" else ocr_full_image(bw_up)

        return {
            "gray": gray,
            "binarised": bw,
            "noise_removed": bw_nb,
            "lines_removed": bw_nl,
            "morph_cleaned": bw_m,
            "upscaled": bw_up,
            "char_boxes": boxes,
            "legacy_result": legacy_text,
            "pipeline_results": pipeline_results,
            "all_candidates": all_candidates,
            "voted_result": voted,
        }
