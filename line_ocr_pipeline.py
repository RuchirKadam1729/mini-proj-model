# line_ocr_pipeline_fixed_v2.py
import os
import csv
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft
from typing import List

# -------------------------------
# Config
# -------------------------------
PAD_W = 1.3            # pad factor along width (text direction)
PAD_H = 1.15           # pad factor along height
MIN_WIDTH_PX = 12      # minimum allowed warped width (reject narrower)
MERGE_VERTICAL_GAP_FACTOR = 0.18
DEBUG_DIR = "results/debug_bad_crops"
OUT_DIR = "results"
TARGET_H = 128
SAVE_OVERLAY = True

# -------------------------------
# Utils
# -------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_overlay_with_indices(img_bgr, boxes, out_path):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes):
        pts = [(float(x), float(y)) for x,y in b.reshape(-1,2)]
        draw.polygon(pts, outline=(255,0,0))
        cx = sum([p[0] for p in pts]) / len(pts)
        cy = sum([p[1] for p in pts]) / len(pts)
        draw.text((cx, cy), str(i), fill=(255,0,0), font=font)
    pil.save(out_path)

# -------------------------------
# deskew_and_clean
# -------------------------------
def deskew_and_clean(cv2img, target_h=TARGET_H):
    if cv2img is None:
        return None
    img = cv2img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Otsu -> ink mask (white)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - th
    coords = cv2.findNonZero(ink)
    if coords is not None and len(coords) > 10:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    kernel_w = max(1, int(gray.shape[1] * 0.45))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.subtract(gray, opened)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(cleaned)

    h0, w0 = clahe_img.shape
    scale = target_h / float(h0) if h0 > 0 else 1.0
    new_w = max(1, int(round(w0 * scale)))
    out = cv2.resize(clahe_img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out_rgb)

# -------------------------------
# robust warp + convex hull fallback
# -------------------------------
def robust_warp_from_quad(img: np.ndarray, quad, pad_w=PAD_W, pad_h=PAD_H, min_width_px=MIN_WIDTH_PX):
    if img is None:
        return None
    pts = np.asarray(quad, dtype=np.float32).reshape(-1,2)
    if pts.size == 0:
        return None

    pts_for_rect = pts
    if pts.shape[0] > 4:
        try:
            hull = cv2.convexHull(pts)
            pts_for_rect = hull.reshape(-1,2)
        except Exception:
            pts_for_rect = pts

    rect = cv2.minAreaRect(pts_for_rect)
    (cx, cy), (w, h), angle = rect
    w, h = float(w), float(h)
    if w == 0 or h == 0:
        return None

    if w < h:
        w, h = h, w
        angle += 90.0

    w = max(1.0, w * pad_w)
    h = max(1.0, h * pad_h)

    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    dst_w = max(1, int(round(w)))
    dst_h = max(1, int(round(h)))
    if dst_w < min_width_px:
        return None
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return warped

# -------------------------------
# fallback axis-aligned padded crop
# -------------------------------
def axis_aligned_padded_crop(img, quad, pad_px=8):
    pts = np.asarray(quad, dtype=np.int32).reshape(-1,2)
    x,y,w,h = cv2.boundingRect(pts)
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(img.shape[1], x + w + pad_px)
    y1 = min(img.shape[0], y + h + pad_px)
    return img[y0:y1, x0:x1]

# -------------------------------
# merge boxes (same as before)
# -------------------------------
def merge_boxes_by_vertical_overlap(boxes: List[np.ndarray], gap_factor=MERGE_VERTICAL_GAP_FACTOR):
    entries = []
    for b in boxes:
        pts = np.asarray(b).reshape(-1,2)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        entries.append({"pts": pts, "x":x, "y":y, "w":w, "h":h})
    if not entries:
        return []
    heights = [e["h"] for e in entries]
    median_h = np.median(heights) if heights else 0.0
    gap_thresh = gap_factor * max(1.0, median_h)
    entries = sorted(entries, key=lambda e: e["y"])
    merged = []
    cur = entries[0].copy()
    for e in entries[1:]:
        gap = e["y"] - (cur["y"] + cur["h"])
        x_overlap = max(0, min(cur["x"]+cur["w"], e["x"]+e["w"]) - max(cur["x"], e["x"]))
        if (gap <= gap_thresh) and (x_overlap > 0 or gap < (0.25 * median_h)):
            combined_pts = np.vstack([cur["pts"], e["pts"]])
            x2, y2, w2, h2 = cv2.boundingRect(combined_pts.astype(np.int32))
            cur = {"pts": combined_pts, "x": x2, "y": y2, "w": w2, "h": h2}
        else:
            merged.append(cur)
            cur = e.copy()
    merged.append(cur)
    return [m["pts"] for m in merged]

# -------------------------------
# artifact detector for warped images
# -------------------------------
def is_artifact(warped_bgr: np.ndarray):
    """Return True if warp looks like a bad strip/stripes artifact."""
    if warped_bgr is None:
        return True
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # too narrow
    if w < MIN_WIDTH_PX or h < 6:
        return True
    # compute median row-diff magnitude (if very small -> many identical rows -> banding)
    row_diffs = np.mean(np.abs(np.diff(gray.astype(np.int16), axis=0)), axis=1)
    col_diffs = np.mean(np.abs(np.diff(gray.astype(np.int16), axis=1)), axis=0)
    row_med = np.median(row_diffs) if row_diffs.size else 255
    col_med = np.median(col_diffs) if col_diffs.size else 255
    # if many rows are nearly identical (med < small threshold) -> artifact
    if row_med < 2.0:
        return True
    # if many columns nearly identical -> very narrow vertical pattern
    if col_med < 2.0 and (w < 40 or (col_med < 1.0)):
        return True
    # otherwise probably ok
    return False

# -------------------------------
# CRAFT wrapper
# -------------------------------
print("Initializing CRAFT detector (pip fork)...")
CRAFT_CUDA = torch.cuda.is_available()
CRAFT_OBJ = Craft(output_dir=None, cuda=CRAFT_CUDA)

def detect_lines(image_path: str):
    pred = CRAFT_OBJ.detect_text(image_path)
    boxes = pred.get("boxes", [])
    norm_boxes = []
    for b in boxes:
        arr = np.asarray(b, dtype=np.float32).reshape(-1,2)
        norm_boxes.append(arr)
    return norm_boxes

# -------------------------------
# Compute stats
# -------------------------------
def compute_box_ratios(boxes):
    ratios = []
    dims = []
    for b in boxes:
        pts = np.asarray(b).reshape(-1,2)
        try:
            hull = cv2.convexHull(pts.astype(np.float32))
            rect = cv2.minAreaRect(hull)
        except Exception:
            rect = cv2.minAreaRect(pts)
        (cx,cy), (w,h), ang = rect
        if w == 0 or h == 0:
            continue
        ratio = min(w,h)/max(w,h)
        ratios.append(ratio)
        dims.append((w,h,ang))
    return np.array(ratios) if ratios else np.array([]), dims

# -------------------------------
# Main pipeline
# -------------------------------
def crop_and_recognize(image_path, boxes, out_dir=OUT_DIR):
    ensure_dir(out_dir)
    ensure_dir(DEBUG_DIR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    if SAVE_OVERLAY:
        save_overlay_with_indices(img, boxes, os.path.join(out_dir, "overlay_boxes.jpg"))

    boxes_merged = merge_boxes_by_vertical_overlap(boxes)
    ratios_before, _ = compute_box_ratios(boxes)
    ratios_after, _ = compute_box_ratios(boxes_merged)
    print(f"Boxes: before={len(boxes)} after_merge={len(boxes_merged)}")
    if ratios_before.size:
        print(f"frac_skinny_before={(ratios_before < 0.2).mean():.3f}, median_ratio={np.median(ratios_before):.3f}")
    if ratios_after.size:
        print(f"frac_skinny_after={(ratios_after < 0.2).mean():.3f}, median_ratio={np.median(ratios_after):.3f}")

    # write box stats CSV
    csv_path = os.path.join(out_dir, "box_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["index","bbox_x","bbox_y","bbox_w","bbox_h","minArea_w","minArea_h","minArea_angle","used_fallback","warp_ok"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading TrOCR on", device)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

    results = []
    for i, quad in enumerate(boxes_merged):
        # log bbox/minArea
        pts = np.asarray(quad).reshape(-1,2).astype(np.float32)
        bx, by, bw, bh = cv2.boundingRect(pts.astype(int))
        try:
            hull = cv2.convexHull(pts)
            (cx,cy), (mw,mh), ang = cv2.minAreaRect(hull)
        except Exception:
            (cx,cy), (mw,mh), ang = cv2.minAreaRect(pts)

        used_fallback = False
        warp_ok = False

        warped = robust_warp_from_quad(img, quad)
        if warped is None or is_artifact(warped):
            # save warped (if exists) for debug
            if warped is not None:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"warped_artifact_{i:03d}.jpg"), warped)
            # try axis-aligned padded crop as fallback
            fallback = axis_aligned_padded_crop(img, quad, pad_px=12)
            used_fallback = True
            if fallback is None or fallback.size == 0:
                print(f"Box {i}: fallback crop empty -> skip")
                with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow([i,bx,by,bw,bh,mw,mh,ang,used_fallback,False])
                continue
            # check fallback artifact just in case
            if is_artifact(fallback):
                # still broken -> save and skip
                cv2.imwrite(os.path.join(DEBUG_DIR, f"fallback_bad_{i:03d}.jpg"), fallback)
                print(f"Box {i}: fallback also looks bad -> saved to debug, skipping")
                with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow([i,bx,by,bw,bh,mw,mh,ang,used_fallback,False])
                continue
            else:
                warp_ok = True
                warped = fallback

        else:
            warp_ok = True

        # deskew and clean warped (expect BGR)
        prepped_pil = deskew_and_clean(warped, target_h=TARGET_H)
        if prepped_pil is None:
            print(f"Box {i}: deskew returned None -> skipping")
            with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow([i,bx,by,bw,bh,mw,mh,ang,used_fallback,False])
            continue

        pixel_values = processor(prepped_pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, num_beams=5, max_length=128, early_stopping=True)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        fname = f"line_{i:03d}"
        prepped_pil.save(os.path.join(OUT_DIR, f"{fname}.png"))
        with open(os.path.join(OUT_DIR, f"{fname}.txt"), "w", encoding="utf-8") as tf:
            tf.write(text)

        with open(csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([i,bx,by,bw,bh,mw,mh,ang,used_fallback,warp_ok])

        results.append((prepped_pil, text))

    # unload craft models
    try:
        CRAFT_OBJ.unload_craftnet_model()
        CRAFT_OBJ.unload_refinenet_model()
    except Exception:
        pass

    # summary
    if results:
        summary_h = len(results) * (TARGET_H + 12)
        summary_w = 900
        summary = Image.new("RGB", (summary_w, summary_h), "white")
        yoff = 0
        draw = ImageDraw.Draw(summary)
        for prepped, text in results:
            preview = prepped.resize((min(400, prepped.width), TARGET_H))
            summary.paste(preview, (4, yoff))
            draw.text((420, yoff + TARGET_H//4), text, fill=(0,0,0))
            yoff += TARGET_H + 12
        summary.save(os.path.join(OUT_DIR, "summary.png"))
        print("Saved summary:", os.path.join(OUT_DIR, "summary.png"))

    print("Done. Recognized:", len(results))
    return results

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", required=True, help="page image path")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out
    ensure_dir(OUT_DIR)
    ensure_dir(DEBUG_DIR)

    boxes = detect_lines(args.page)
    print("Detected boxes:", len(boxes))
    crop_and_recognize(args.page, boxes, out_dir=OUT_DIR)
