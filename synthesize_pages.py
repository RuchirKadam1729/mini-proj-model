# synthesize_pages.py
import os, glob, random, cv2, numpy as np
from PIL import Image
os.makedirs("gnhk_dataset/synth_pages", exist_ok=True)
crop_list = glob.glob("gnhk_dataset/train_processed/*jpg") + glob.glob("gnhk_dataset/page_line_crops/*jpg")
crop_list = [p for p in crop_list if os.path.exists(p)]
bg = np.full((3508, 2480, 3), 240, dtype=np.uint8)  # A4-like light paper

def make_page(idx, n_lines=20):
    canvas = bg.copy()
    h, w = canvas.shape[:2]
    y = 100
    annotations = []
    for i in range(n_lines):
        p = cv2.imread(random.choice(crop_list))
        if p is None: continue
        scale = 40.0 / max(p.shape[0], 1)  # scale to ~40 px height then resize later
        new_h = int(p.shape[0] * scale)
        new_w = int(p.shape[1] * scale)
        p = cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = random.randint(60, 200)
        if y + new_h >= h - 200:
            break
        # paste with alpha-like blending
        canvas[y:y+new_h, x:x+new_w] = p
        annotations.append((x,y,x+new_w,y+new_h))
        y += new_h + random.randint(8, 30)
    out_img = f"gnhk_dataset/synth_pages/page_synth_{idx:04d}.jpg"
    cv2.imwrite(out_img, canvas)
    # write annotations as axis-aligned boxes
    with open(out_img.replace('.jpg','.txt'),'w') as f:
        for a in annotations:
            f.write(','.join(map(str,a)) + '\n')

for i in range(100):
    make_page(i, n_lines=random.randint(12,30))
print("synth pages done")
