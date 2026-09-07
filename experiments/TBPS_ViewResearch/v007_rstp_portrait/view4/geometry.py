"""Immutable portrait geometry for this independent experiment snapshot."""
INPUT_SIZE = (384, 128)
# Preserve the square baseline's relative aspect jitter around the new W/H.
CROP_RATIO = (0.75 * INPUT_SIZE[1] / INPUT_SIZE[0],
              (4. / 3.) * INPUT_SIZE[1] / INPUT_SIZE[0])


def check_geometry(cfg):
    value = cfg.get('input_resolution')
    if not isinstance(value, (list, tuple)) or tuple(value) != INPUT_SIZE:
        raise ValueError('V007 requires explicit input_resolution [384,128]')
