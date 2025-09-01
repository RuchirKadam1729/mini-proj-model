import os, glob, cv2, numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft
import torch
# -------------------------------
# CONFIG
# -------------------------------
PAGE_DIR = "gnhk_dataset/full_pages"
USE_CRAFT = True  # if False, use existing polygons / crops
CROP_OUT_DIR = "gnhk_dataset/temp_crops"
os.makedirs(CROP_OUT_DIR, exist_ok=True)

# TrOCR
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

# -------------------------------
# LINE DETECTION
# -------------------------------
def detect_lines_with_craft(img_path):
    craft = Craft(output_dir=CROP_OUT_DIR, crop_type="poly", cuda=False)
    pred = craft.detect_text(img_path)
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    crops = []
    for box in pred["boxes"]:
        # crop box (rotated or poly)
        pts = np.array(box).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        img = cv2.imread(img_path)
        crop = img[y:y+h, x:x+w]
        crops.append(crop)
    return crops

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess_crop(cv2img, target_h=128):
    if cv2img is None or cv2img.size == 0:
        return None
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    h, w = gray.shape
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(rgb)

# -------------------------------
# OCR
# -------------------------------
def ocr_line(pil_img):
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# -------------------------------
# FULL PAGE PIPELINE
# -------------------------------
def ocr_page(img_path):
    if USE_CRAFT:
        crops = detect_lines_with_craft(img_path)
    else:
        # fallback: read from precomputed page_line_crops_corrected/
        base = os.path.splitext(os.path.basename(img_path))[0]
        crops = []
        crop_files = sorted(glob.glob(f"gnhk_dataset/page_line_crops_corrected/{base}_*.jpg"))
        for cf in crop_files:
            crop_img = cv2.imread(cf)
            if crop_img is not None:
                crops.append(crop_img)

    full_text = []
    for c in crops:
        pil = preprocess_crop(c)
        if pil is None: continue
        text = ocr_line(pil)
        full_text.append(text)
    return "\n".join(full_text)

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    pages = glob.glob(os.path.join(PAGE_DIR, "*.jpg")) + glob.glob(os.path.join(PAGE_DIR, "*.png"))
    for p in pages:
        text = ocr_page(p)
        print(f"\n==== {os.path.basename(p)} ====\n{text}\n")
