# src/normalize_crops.py
import os, glob, cv2, numpy as np
from math import atan2, degrees
from PIL import Image

IN_DIR = "gnhk_dataset/page_line_crops"
OUT_DIR = "gnhk_dataset/page_line_crops_corrected"
os.makedirs(OUT_DIR, exist_ok=True)

def rect_box_and_angle_from_points(pts):
    """Given 4x2 points (float), pick the longer edge and compute its angle in degrees."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4,2)
    # ensure order: cv2.boxPoints gives a consistent order but we don't rely on sign
    edges = [pts[(i+1)%4] - pts[i] for i in range(4)]
    lens = [np.linalg.norm(e) for e in edges]
    idx = int(np.argmax(lens))  # index of longest edge
    vec = edges[idx]
    ang = degrees(atan2(vec[1], vec[0]))  # angle in degrees of longest edge
    return ang

def crop_and_normalize_from_poly(img, poly, pad=8, target_h=128):
    """
    poly: list of 8 ints [x0,y0,...,x3,y3] (or array (4,2))
    returns: normalized PIL.Image (RGB) or None if crop empty
    """
    pts = np.array(poly, dtype=np.float32).reshape(4,2)
    rect = cv2.minAreaRect(pts)   # ((cx,cy),(w,h),angle) - angle here is ambiguous; we'll compute robust angle
    box = cv2.boxPoints(rect)     # 4x2
    ang = rect_box_and_angle_from_points(box)  # angle of longest edge (degrees)
    # Rotate the whole image by -ang so the longest edge becomes horizontal
    (h_img, w_img) = img.shape[:2]
    M = cv2.getRotationMatrix2D((rect[0][0], rect[0][1]), -ang, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # After rotation, compute the axis-aligned bounding box of the rotated polygon
    pts_rot = cv2.transform(np.array([pts]), M)[0]  # transformed polygon points
    x_min = int(np.min(pts_rot[:,0]) - pad)
    y_min = int(np.min(pts_rot[:,1]) - pad)
    x_max = int(np.max(pts_rot[:,0]) + pad)
    y_max = int(np.max(pts_rot[:,1]) + pad)
    x_min = max(0, x_min); y_min = max(0, y_min)
    x_max = min(w_img-1, x_max); y_max = min(h_img-1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = rotated[y_min:y_max, x_min:x_max]

    # If crop is much taller than wide, rotate 90 degrees to make it horizontal
    ch, cw = crop.shape[:2]
    if ch > cw:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        ch, cw = crop.shape[:2]

    # contrast & resize to target height
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    scale = target_h / float(rgb.shape[0])
    new_w = max(1, int(rgb.shape[1] * scale))
    rgb = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(rgb)
    return pil

def find_poly_for_crop(fname):
    """
    Tries to infer a polygon from filename when available like <page>_line<idx>.jpg
    If you stored polygons separately (page_annotations), you should adapt this function.
    For now we will just read the crop image itself (no polygon).
    """
    # fallback: we don't need poly to normalize an existing crop; we'll treat the whole image as crop
    return None

def normalize_existing_crop(img_path, out_path, target_h=128):
    img = cv2.imread(img_path)
    if img is None:
        print("can't read", img_path); return False
    h,w = img.shape[:2]
    # If image is already horizontal-ish and reasonable size, just run CLAHE+resize
    if w >= h:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        scale = target_h / float(rgb.shape[0])
        new_w = max(1, int(rgb.shape[1] * scale))
        rgb = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        Image.fromarray(rgb).save(out_path)
        return True
    # if tall: rotate to make it horizontal
    else:
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        scale = target_h / float(rgb.shape[0])
        new_w = max(1, int(rgb.shape[1] * scale))
        rgb = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        Image.fromarray(rgb).save(out_path)
        return True

# MAIN: try to normalize existing crops. If there was a polygon available for each crop,
# you'd want to call crop_and_normalize_from_poly on the page image instead.
all_files = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")) + glob.glob(os.path.join(IN_DIR, "*.png")))
print("found", len(all_files), "crops to normalize")
count = 0
for p in all_files:
    out_p = os.path.join(OUT_DIR, os.path.basename(p))
    ok = normalize_existing_crop(p, out_p, target_h=128)
    if ok:
        count += 1
print("wrote", count, "normalized crops to", OUT_DIR)
