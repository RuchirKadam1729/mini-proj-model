# eval_e2e.py
import numpy as np, cv2, glob, os, math
import editdistance
from collections import namedtuple

def aabb_from_poly(box):
    pts = np.array(box).reshape(4,2)
    x0 = int(pts[:,0].min()); y0 = int(pts[:,1].min())
    x1 = int(pts[:,0].max()); y1 = int(pts[:,1].max())
    return (x0,y0,x1,y1)

def iou(a,b):
    ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
    interx0 = max(ax0,bx0); intery0 = max(ay0,by0)
    interx1 = min(ax1,bx1); intery1 = min(ay1,by1)
    iw = max(0, interx1-interx0); ih = max(0, intery1-intery0)
    inter = iw*ih
    areaA = (ax1-ax0)*(ay1-ay0); areaB=(bx1-bx0)*(by1-by0)
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

# example usage: supply mapping of page->GT lines from page_annotations/*.txt and page->preds from pipeline
# Here, for brevity, this script expects two directories:
# gt_dir: files named <page>.txt with each line: x0,y0,x1,y1,x2,y2,x3,y3
# pred_dir: files named <page>.pred.txt with each line: x0,y0,x1,y1,x2,y2,x3,y3\ttext (or you can adapt)
gt_dir = "gnhk_dataset/page_annotations"
pred_dir = "gnhk_dataset/predictions"  # make by saving inference results as described below

page_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(gt_dir,"*.txt"))]

total_chars = 0
total_errors = 0
matched = 0
for p in page_files:
    gt_path = os.path.join(gt_dir, p+".txt")
    pred_path = os.path.join(pred_dir, p+".pred.txt")
    if not os.path.exists(pred_path): continue
    gtl = [line.strip().split(',') for line in open(gt_path,'r',encoding='utf-8') if line.strip()]
    gtbboxes = [list(map(int, l[:8])) for l in gtl]
    preds = []
    for line in open(pred_path,'r',encoding='utf-8'):
        sp = line.strip().split('\t')
        coords = list(map(int, sp[0].split(',')))
        text = sp[1] if len(sp)>1 else ''
        preds.append((coords,text))
    # match preds -> gt by IoU (greedy)
    gt_used = set()
    for coords,text in preds:
        pb = aabb_from_poly(coords)
        best_iou = 0; best_idx = -1
        for i,gb in enumerate(gtbboxes):
            if i in gt_used: continue
            gb_a = aabb_from_poly(gb)
            val = iou(pb, gb_a)
            if val>best_iou:
                best_iou=val; best_idx=i
        if best_iou >= 0.3 and best_idx>=0:
            gt_used.add(best_idx)
            # find gt text — optional: you may have GT text elsewhere; for now we compute char edit distance if you have a GT mapping
            # Here we assume there is a page-level mapping from line-index to text; adapt if you have one.
            # For demo, skip text match
            matched += 1
        else:
            # unmatched pred (false positive)
            pass

print("pages evaluated:", len(page_files), "pred matches:", matched)
