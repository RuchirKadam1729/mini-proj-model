# inference_from_annotations.py
import os, glob, cv2, numpy as np, math
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Paths - adjust if your layout differs
PAGE_IMG_DIR = "gnhk_dataset/train_data/train"        # where page .jpg/.png are
PAGE_ANNOT_DIR = "gnhk_dataset/page_annotations"      # *.txt produced by make_line_polys.py
OUT_PRED_DIR = "gnhk_dataset/predictions"
os.makedirs(OUT_PRED_DIR, exist_ok=True)

# Load TrOCR (change to your tuned model path if you have one)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(device)

def read_poly_txt(path):
    """Each line: x0,y0,x1,y1,x2,y2,x3,y3"""
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            if len(parts) < 8: continue
            coords = list(map(int, parts[:8]))
            out.append(coords)
    return out

def crop_rotated_from_poly(img, poly, pad=8):
    pts = np.array(poly, dtype=np.float32).reshape(4,2)
    rect = cv2.minAreaRect(pts)   # ((cx,cy),(w,h),angle)
    (cx,cy),(w,h),ang = rect
    w = max(1,int(w + pad*2)); h = max(1,int(h + pad*2))
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    h_img,w_img = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x = int(cx - w/2); y = int(cy - h/2)
    x = max(0,x); y = max(0,y); x2 = min(w_img, x + w); y2 = min(h_img, y + h)
    return rotated[y:y2, x:x2]

def preprocess_for_trocr(cv2img, target_h=128):
    if cv2img is None or cv2img.size==0: return None
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    h = rgb.shape[0]; scale = target_h / float(h)
    new_w = max(1, int(rgb.shape[1]*scale))
    rgb = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rgb)

def infer_page(page_base):
    # page_base is filename without ext, e.g. eng_EU_091
    # find page image
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(PAGE_IMG_DIR, page_base + ext)
        if os.path.exists(p):
            page_path = p
            break
    else:
        print("page image not found for", page_base)
        return []

    img = cv2.imread(page_path)
    poly_file = os.path.join(PAGE_ANNOT_DIR, page_base + ".txt")
    if not os.path.exists(poly_file):
        print("annotation missing for", page_base)
        return []

    polys = read_poly_txt(poly_file)
    crops = []
    bboxes = []
    for poly in polys:
        crop = crop_rotated_from_poly(img, poly)
        pil = preprocess_for_trocr(crop)
        if pil is None: continue
        crops.append(pil)
        bboxes.append(poly)

    if not crops:
        return []

    pixel_values = processor(images=crops, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(pixel_values, num_beams=5, max_length=256)
    preds = processor.batch_decode(outputs, skip_special_tokens=True)
    return list(zip(bboxes, preds))

if __name__ == "__main__":
    # process all page_annotation files (or limit to a subset)
    files = glob.glob(os.path.join(PAGE_ANNOT_DIR, "*.txt"))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        res = infer_page(base)
        # write predictions to file: coords<TAB>text
        outp = os.path.join(OUT_PRED_DIR, base + ".pred.txt")
        with open(outp, 'w', encoding='utf-8') as fo:
            for coords, text in res:
                fo.write(','.join(map(str,coords)) + "\t" + text + "\n")
        print("wrote", outp, "lines:", len(res))
