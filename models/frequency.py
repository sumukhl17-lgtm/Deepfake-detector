import numpy as np
from PIL import Image


FEATURE_LENGTH = 128


def _as_grayscale_array(face_crop: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(face_crop, Image.Image):
        return np.asarray(face_crop.convert("L"), dtype=np.float32)
    array = np.asarray(face_crop)
    if array.ndim == 3:
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = np.dot(array[..., :3], np.array([0.299, 0.587, 0.114]))
    if array.ndim != 2:
        raise ValueError("face_crop must be a grayscale or RGB image")
    return array.astype(np.float32, copy=False)


def extract_frequency_features(face_crop: Image.Image | np.ndarray) -> np.ndarray:
    grayscale = _as_grayscale_array(face_crop)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grayscale))))
    y, x = np.indices(spectrum.shape)
    center = np.array([(spectrum.shape[1] - 1) / 2, (spectrum.shape[0] - 1) / 2])
    radii = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    radial_sum = np.bincount(radii.ravel(), weights=spectrum.ravel())
    radial_count = np.bincount(radii.ravel())
    radial_profile = radial_sum / np.maximum(radial_count, 1)
    if radial_profile.size == FEATURE_LENGTH:
        return radial_profile.astype(np.float32)
    source = np.linspace(0.0, 1.0, radial_profile.size)
    target = np.linspace(0.0, 1.0, FEATURE_LENGTH)
    return np.interp(target, source, radial_profile).astype(np.float32)
