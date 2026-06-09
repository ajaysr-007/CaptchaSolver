import numpy as np

import captcha_solver.solver as solver_module
from captcha_solver import (
    CaptchaSolver,
    binarise,
    load_and_normalise,
    remove_noise_blobs,
    segment_characters,
    enhance_contrast,
    median_denoise,
    remove_lines_by_components,
    vote_on_results,
    crop_yellow_line,
)


def test_load_and_normalise_inverts_dark_image():
    dark = np.zeros((10, 10, 3), dtype=np.uint8)
    result = load_and_normalise(dark)
    assert result.dtype == np.uint8
    assert result.shape == (10, 10)
    assert np.all(result == 255)


def test_binarise_returns_binary_image():
    gray = np.array([[0, 255], [128, 100]], dtype=np.uint8)
    for method in ("global", "otsu", "adaptive", "sauvola"):
        result = binarise(gray, method=method)
        assert set(np.unique(result)).issubset({0, 255})


def test_remove_noise_blobs_removes_small_objects():
    image = np.full((50, 50), 255, dtype=np.uint8)
    image[10:30, 10:30] = 0
    image[5, 5] = 0
    cleaned = remove_noise_blobs(image, min_area=10, min_height=1)
    assert cleaned[5, 5] == 255
    assert np.all(cleaned[10:30, 10:30] == 0)


def test_segment_characters_merges_overlapping_regions():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:40, 10:30] = 255
    image[10:40, 25:45] = 255
    up = solver_module.upscale(image, scale=1)
    boxes = segment_characters(up, min_area=10, min_height=10, min_width=5, merge_overlap_px=5)
    assert len(boxes) == 1
    assert boxes[0][0] <= 10
    assert boxes[0][2] >= 30


def test_enhance_contrast_returns_same_shape():
    gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    result = enhance_contrast(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_median_denoise_returns_same_shape():
    gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    result = median_denoise(gray, ksize=3)
    assert result.shape == gray.shape


def test_vote_on_results_picks_majority():
    candidates = ["ABCDE", "ABCDE", "XBCDE", "ABCDF"]
    result = vote_on_results(candidates, expected_len=5)
    assert result == "ABCDE"


def test_vote_on_results_position_voting():
    candidates = ["ABCDE", "XBCDE", "XBCDE"]
    result = vote_on_results(candidates, expected_len=5)
    assert result == "XBCDE"


def test_vote_on_results_no_exact_length():
    candidates = ["ABC", "ABCD"]
    result = vote_on_results(candidates, expected_len=5)
    assert result in candidates


def test_vote_on_results_empty():
    assert vote_on_results([]) == ""


def test_crop_yellow_line_no_yellow():
    # White image - no yellow to crop
    img = np.full((50, 100, 3), 255, dtype=np.uint8)
    result = crop_yellow_line(img)
    assert np.array_equal(result, img)


def test_remove_lines_by_components_preserves_characters():
    # Create an image with a character-like block
    bw = np.full((50, 100), 255, dtype=np.uint8)
    bw[10:40, 20:35] = 0  # character-like: tall, narrow
    result = remove_lines_by_components(bw)
    assert np.all(result[10:40, 20:35] == 0)  # character preserved


def test_solver_uses_voting_by_default():
    solver = CaptchaSolver()
    assert solver.use_voting is True


def test_solver_prefers_longer_text(monkeypatch):
    solver = CaptchaSolver(use_voting=False)
    monkeypatch.setattr(solver_module, "load_and_normalise", lambda source: np.zeros((10, 10), dtype=np.uint8))
    monkeypatch.setattr(solver_module, "binarise", lambda gray, **kwargs: np.zeros_like(gray))
    monkeypatch.setattr(solver_module, "remove_noise_blobs", lambda bw, min_area, min_height: bw)
    monkeypatch.setattr(solver_module, "remove_lines", lambda bw, **kwargs: bw)
    monkeypatch.setattr(solver_module, "morphological_cleanup", lambda bw: bw)
    monkeypatch.setattr(solver_module, "upscale", lambda bw, scale: bw)
    monkeypatch.setattr(solver_module, "segment_characters", lambda bw, min_area, min_height: [(0, 0, 1, 1)])
    monkeypatch.setattr(solver_module, "ocr_per_character", lambda bw, boxes: "A")
    monkeypatch.setattr(solver_module, "ocr_full_image", lambda bw: "AB")
    assert solver.solve("ignored") == "AB"
