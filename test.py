python - <<'EOF'
from craft_text_detector import Craft

# init
craft = Craft(output_dir="out", crop_type="poly", cuda=False)

# run test on an image (replace with one of your scans)
prediction_result = craft.detect_text("page_line_crops_corrected/0001.jpg")

# cleanup
craft.unload_craftnet_model()
craft.unload_refinenet_model()

print("Detected regions:", len(prediction_result["boxes"]))
EOF
