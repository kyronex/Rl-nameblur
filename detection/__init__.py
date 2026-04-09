# detection/__init__.py
from .detect import detect_plates, ncc_match
from .tools import write_circles , write_rects , get_color
from .mask import compute_white_mask, compute_sobel_interior_unified, refine_and_merge ,saturation_variance_mask
from .boxes import process_channel