"""Preprocessing matched to the original notebook; local model loading only."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent

@lru_cache(maxsize=2)
def load_model(name):
    path = (BASE / name).resolve()
    if not path.is_file():
        raise RuntimeError(f'Missing model weights: {path}')
    try:
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(path, compile=False)
    except Exception as exc:
        raise RuntimeError(f'Could not load compatible model weights: {path.name}') from exc

def image_pre(upload):
    try:
        with Image.open(upload) as image:
            if image.format not in {'PNG', 'JPEG'}:
                raise ValueError('Only PNG and JPEG image contents are supported.')
            if image.width * image.height > 16000000:
                raise ValueError('Use an image with no more than 16 million pixels.')
            image = ImageOps.grayscale(image)
            image = image.resize((128, 128), resample=Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('The file is not a readable image.') from exc
    # Preserve training polarity.
    # Training used raw 0-255 pixel values.
    return pixels.reshape((1, 128, 128, 1))

def scalar(model, data):
    try:
        output = np.asarray(model(data, training=False)).reshape(-1)
    except Exception as exc:
        raise RuntimeError('The model could not process this input.') from exc
    if output.size != 1 or not np.isfinite(output).all():
        raise RuntimeError('Invalid model output.')
    return float(output[0])

def predict(data):
    probability = scalar(load_model('deepfake.h5'), data)
    if not 0 <= probability <= 1:
        raise RuntimeError('Invalid classifier output range.')
    return int(probability >= 0.5)
