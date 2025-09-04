import cv2
from PIL import Image
from craft_text_detector import Craft
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
# ---- CONFIG ----
PAGE_IMAGE = "image.png"  # path to your new page image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- INIT MODELS ----
# CRAFT text detector
craft = Craft(output_dir=None, crop_type="poly", cuda=torch.cuda.is_available())

# TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(DEVICE)

# ---- LOAD IMAGE ----
img = cv2.imread(PAGE_IMAGE)
if img is None:
    raise FileNotFoundError(f"Image not found: {PAGE_IMAGE}")



# ---- DETECT TEXT REGIONS ----
prediction_result = craft.detect_text(PAGE_IMAGE)

boxes = prediction_result["boxes"]  # list of polygons, each [[x0,y0],...,[x3,y3]]
if not boxes:
    print("No text regions detected.")
else:
    print(f"Detected {len(boxes)} text regions.")

# diagnostics
from PIL import Image, ImageDraw
img1 = Image.open('image.png')
# draw boxes (boxes from craft.detect_text result)
draw = ImageDraw.Draw(img1)
for b in boxes: draw.polygon(b, outline='red')
img1.show()
# and save a few crops for inspection

# ---- CROP AND RUN OCR ----
for i, box in enumerate(boxes):
    pts = cv2.convexHull(np.array(box).astype(int)).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h_img, w_img = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x = int(cx - w / 2); y = int(cy - h / 2)
    x2 = int(cx + w / 2); y2 = int(cy + h / 2)
    crop = rotated[max(0, y):min(h_img, y2), max(0, x):min(w_img, x2)]

    # preprocess for TrOCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(rgb)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # remove horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    horiz = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    mask = cv2.subtract(g, horiz)
    # convert to 3-channel
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # resize
    h_target = 128
    scale = h_target / img.shape[0]
    w = max(1, int(img.shape[1]*scale))
    img = cv2.resize(img, (w, h_target), interpolation=cv2.INTER_CUBIC)

    # OCR
    pixel_values = processor(images=[pil_img], return_tensors="pt").pixel_values.to(DEVICE)
    outputs = model.generate(pixel_values, num_beams=5, max_length=256)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"[Line {i+1}] {text}")

# ---- CLEANUP ----
craft.unload_craftnet_model()
craft.unload_refinenet_model()
