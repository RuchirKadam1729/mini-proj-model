# make_line_polys.py
import json, os, glob
import numpy as np
import cv2
from collections import defaultdict

json_dir = "gnhk_dataset/train_data/train"   # adjust: folder with page jsons
out_anno_dir = "gnhk_dataset/page_annotations"  # output polygons per page
out_crops_dir = "gnhk_dataset/page_line_crops" # optional save of crops (axis aligned)
os.makedirs(out_anno_dir, exist_ok=True)
os.makedirs(out_crops_dir, exist_ok=True)

def points_from_poly_dict(polydict):
    # JSON polygon format uses x0,y0,x1,y1,... as keys
    pts = []
    i = 0
    while True:
        kx = f"x{i}"
        ky = f"y{i}"
        if kx in polydict and ky in polydict:
            pts.append([polydict[kx], polydict[ky]])
            i += 1
        else:
            break
    return pts

def rotated_rect_from_points(allpts):
    arr = np.array(allpts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)   # (cx,cy),(w,h),angle
    box = cv2.boxPoints(rect)     # 4 x 2
    box = box.astype(int).reshape(-1).tolist()
    return box, rect

def crop_rotated_rect(img, rect, pad=8):
    # rect: ((cx,cy),(w,h),angle)
    (cx,cy),(w,h),ang = rect
    w = max(1,int(w+pad*2))
    h = max(1,int(h+pad*2))
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    h_img,w_img = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x = int(cx - w/2); y = int(cy - h/2)
    x2 = int(x + w); y2 = int(y + h)
    x = max(0,x); y = max(0,y); x2=min(w_img,x2); y2=min(h_img,y2)
    return rotated[y:y2, x:x2]

# iterate all page jsons
for jp in glob.glob(os.path.join(json_dir, "*.json")):
    base = os.path.splitext(os.path.basename(jp))[0]
    with open(jp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data is an array of word entries; each has 'polygon' dict and 'line_idx'
    lines = defaultdict(list)
    for item in data:
        if 'line_idx' not in item or 'polygon' not in item: 
            continue
        li = item['line_idx']
        poly = item['polygon']
        pts = points_from_poly_dict(poly)
        if len(pts) == 0:
            continue
        lines[li].extend(pts)

    if not lines:
        print("no line data for", base); continue

    # produce a txt file with polygon coords per line (4-point rotated box)
    out_txt = os.path.join(out_anno_dir, base + ".txt")
    with open(out_txt, 'w', encoding='utf-8') as out:
        # sort by line_idx
        for li in sorted(lines.keys()):
            pts = lines[li]
            box, rect = rotated_rect_from_points(pts)
            # write 8 ints: x0,y0,x1,y1,x2,y2,x3,y3
            out.write(','.join(map(str, box)) + '\n')

    # optional: produce crops (rotated rect crops)
    page_img_jpg = os.path.join(json_dir, base + ".jpg")
    if not os.path.exists(page_img_jpg):
        # sometimes image ext is .png or present in other dir; try jpg first then png
        page_img_jpg = os.path.join(json_dir, base + ".png")
    if os.path.exists(page_img_jpg):
        img = cv2.imread(page_img_jpg)
        for li in sorted(lines.keys()):
            pts = lines[li]
            box, rect = rotated_rect_from_points(pts)
            crop = crop_rotated_rect(img, rect, pad=8)
            outp = os.path.join(out_crops_dir, f"{base}_line{li}.jpg")
            cv2.imwrite(outp, crop)
    else:
        # image missing — skip crops
        pass

print("done")
