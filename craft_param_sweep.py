# craft_param_sweep.py
import os, cv2
from main import get_prediction   # your function
# ensure craft_net, refine_net are created/loaded in your environment
# replace these two lines with however you load them (you likely already do this in main)
from craft_text_detector import Craft
craft_wrapper = Craft(output_dir=None, refine_net=True)   # lightweight wrapper
craft_net = craft_wrapper.craft_net
refine_net = craft_wrapper.refine_net

def draw_boxes_and_save(image_path, boxes, out_path):
    img = cv2.imread(image_path)
    for b in boxes:
        pts = b.reshape(-1,2).astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
    cv2.imwrite(out_path, img)

page = "gnhk_dataset/train_data/train/eng_EU_091.jpg"  # swap with your sample page
out_dir = "debug_craft"
os.makedirs(out_dir, exist_ok=True)

param_sets = [
    {"long_size":1280, "text_threshold":0.7, "link_threshold":0.4, "low_text":0.4},
    {"long_size":1600, "text_threshold":0.75,"link_threshold":0.45,"low_text":0.45},
    {"long_size":2000, "text_threshold":0.6, "link_threshold":0.3, "low_text":0.2},
    {"long_size":1200, "text_threshold":0.8, "link_threshold":0.5, "low_text":0.5},
]

for i,ps in enumerate(param_sets):
    print("running set", i, ps)
    res = get_prediction(page, craft_net, refine_net,
                         text_threshold=ps["text_threshold"],
                         link_threshold=ps["link_threshold"],
                         low_text=ps["low_text"],
                         cuda=False,
                         long_size=ps["long_size"],
                         poly=True)
    boxes = res["boxes"]
    outimg = os.path.join(out_dir, f"page_debug_set{i}_ls{ps['long_size']}_tt{ps['text_threshold']}.jpg")
    draw_boxes_and_save(page, boxes, outimg)
    print("wrote", outimg)
