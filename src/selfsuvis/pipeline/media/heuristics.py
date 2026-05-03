
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
except (ImportError, ModuleNotFoundError):
    ssim = None

from selfsuvis.pipeline.core import settings

HIST_BINS = 64
TILE_RESIZE = 64
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200


def downsample_gray(img: np.ndarray, size: int = TILE_RESIZE) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def blur_laplacian_var(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mean_intensity(gray: np.ndarray) -> float:
    return float(gray.mean())


def histogram_diff(a: np.ndarray, b: np.ndarray) -> float:
    hist_a = cv2.calcHist([a], [0], None, [HIST_BINS], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [HIST_BINS], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    diff = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    return float(diff)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def ssim_diff(a: np.ndarray, b: np.ndarray) -> float:
    if ssim is None:
        return 0.0
    score = ssim(a, b)
    return float(1.0 - score)


def phase_corr_align(prev_small: np.ndarray, curr_small: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    shift, response = cv2.phaseCorrelate(prev_small, curr_small)
    dx, dy = shift
    if response < settings.PHASECORR_MIN_RESPONSE:
        return curr_small, dx, dy, response
    if abs(dx) > settings.STAB_MAX_SHIFT or abs(dy) > settings.STAB_MAX_SHIFT:
        return curr_small, dx, dy, response
    M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(curr_small, M, (curr_small.shape[1], curr_small.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, dx, dy, response


def frame_quality_ok(frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if blur_laplacian_var(gray) < settings.BLUR_LAPL_VAR_MIN_FRAME:
        return False
    mean_int = mean_intensity(gray)
    if mean_int < settings.MEAN_INTENSITY_MIN or mean_int > settings.MEAN_INTENSITY_MAX:
        return False
    return True


def tile_quality_ok(tile_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    if blur_laplacian_var(gray) < settings.BLUR_LAPL_VAR_MIN_TILE:
        return False
    mean_int = mean_intensity(gray)
    if mean_int < settings.MEAN_INTENSITY_MIN or mean_int > settings.MEAN_INTENSITY_MAX:
        return False
    if tile_std(gray) < settings.TILE_STD_MIN:
        return False
    if tile_entropy(gray) < settings.TILE_ENTROPY_MIN:
        return False
    if sky_haze_suppress(tile_bgr, gray):
        return False
    return True


def sky_haze_suppress(tile_bgr: np.ndarray, gray: np.ndarray) -> bool:
    b, g, r = cv2.split(tile_bgr)
    blue_ratio = float(np.mean((b > 1.15 * r) & (b > 1.15 * g)))
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edge_density = float(np.mean(edges > 0))
    return blue_ratio > settings.SKY_BLUE_RATIO_MAX and edge_density < settings.EDGE_DENSITY_MIN


def tile_std(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (TILE_RESIZE, TILE_RESIZE), interpolation=cv2.INTER_AREA)
    return float(np.std(small))


def tile_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    ent = -np.sum(hist * np.log2(hist + 1e-8))
    return float(ent)


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    return float(np.mean(edges > 0))


def motion_score(prev_small: np.ndarray, curr_small: np.ndarray) -> float:
    return mean_abs_diff(prev_small, curr_small)
