# import os
# import xml.etree.ElementTree as ET

# XML_FILE = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Car/annotations.xml"
# OUTPUT_DIR = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Car/"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Classes in your dataset
# classes = ["car", "minivan"]

# tree = ET.parse(XML_FILE)
# root = tree.getroot()

# # Get image size from <meta> section
# meta = root.find("meta").find("task").find("original_size")
# IMG_W = int(meta.find("width").text)
# IMG_H = int(meta.find("height").text)


# # Dictionary: frame → list of YOLO lines
# frames = {}

# # Iterate over tracks
# for track in root.findall("track"):
#     label = track.get("label")
#     cls_id = classes.index(label)

#     for box in track.findall("box"):
#         frame_id = int(box.get("frame"))
#         outside = int(box.get("outside"))

#         # Skip if object is outside frame
#         if outside == 1:
#             continue

#         xtl = float(box.get("xtl"))
#         ytl = float(box.get("ytl"))
#         xbr = float(box.get("xbr"))
#         ybr = float(box.get("ybr"))

#         # Convert to YOLO format
#         x_center = ((xtl + xbr) / 2) / IMG_W
#         y_center = ((ytl + ybr) / 2) / IMG_H
#         w = (xbr - xtl) / IMG_W
#         h = (ybr - ytl) / IMG_H

#         line = f"{cls_id} {x_center} {y_center} {w} {h}"

#         if frame_id not in frames:
#             frames[frame_id] = []
#         frames[frame_id].append(line)

# # Write YOLO TXT files
# for frame_id, yolo_lines in frames.items():
#     if frame_id >= 300:
#         break
#     txt_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.txt")
#     with open(txt_path, "w") as f:
#         f.write("\n".join(yolo_lines))

# print("Conversion complete! YOLO labels written to /labels/")


# import os
# import re

# # Folder containing multiple Penn-Fudan text annotation files
# ANNOT_DIR = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Pedestrians"

# # Folder where YOLO labels will be saved
# OUTPUT_DIR = os.path.join(ANNOT_DIR, "labels")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# class_id = 0  # "person" for YOLO

# for filename in os.listdir(ANNOT_DIR):
#     if not filename.endswith(".txt"):
#         continue

#     input_path = os.path.join(ANNOT_DIR, filename)

#     with open(input_path, "r") as f:
#         data = f.read()

#     # Extract image size
#     size_match = re.search(r'Image size.*:\s*(\d+)\s*x\s*(\d+)', data)
#     if not size_match:
#         print(f"Skipping {filename} — no image size found")
#         continue

#     img_w = int(size_match.group(1))
#     img_h = int(size_match.group(2))

#     # Extract all bounding boxes
#     boxes = re.findall(
#         r'Bounding box.*:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
#         data
#     )

#     if not boxes:
#         print(f"No bounding boxes found in {filename}")
#         continue

#     yolo_lines = []

#     for (xmin, ymin, xmax, ymax) in boxes:
#         xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))

#         # YOLO conversion
#         x_center = (xmin + xmax) / 2 / img_w
#         y_center = (ymin + ymax) / 2 / img_h
#         w = (xmax - xmin) / img_w
#         h = (ymax - ymin) / img_h

#         yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

#     # Save YOLO file
#     output_path = os.path.join(
#         OUTPUT_DIR,
#         filename.replace(".txt", "_yolo.txt")
#     )

#     with open(output_path, "w") as f:
#         f.write("\n".join(yolo_lines))

#     print(f"Converted: {filename} → labels/{os.path.basename(output_path)}")

# print("\nConversion completed for all files!")






import os
import re
import shutil
import random

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASE_PATH = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset"

DATASETS = {
    "Car": {"path": f"{BASE_PATH}/Car", "class_id": 0},
    "Cycle": {"path": f"{BASE_PATH}/Cycle", "class_id": 1},
    "Pedestrian": {"path": f"{BASE_PATH}/Pedestrians", "class_id": 2}
}

OUTPUT_IMAGES = f"{BASE_PATH}/images"
OUTPUT_LABELS = f"{BASE_PATH}/labels"

TRAIN_SPLIT = 0.9

CLASSES = ["car", "cycle", "person"]

# Create structure
for folder in [OUTPUT_IMAGES, OUTPUT_LABELS]:
    os.makedirs(os.path.join(folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(folder, "val"), exist_ok=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def convert_pedestrian_annotation(txt_content, class_id):
    # Extract image size
    size_match = re.search(r'Image size.*:\s*(\d+)\s*x\s*(\d+)', txt_content)
    if not size_match:
        return None

    img_w = int(size_match.group(1))
    img_h = int(size_match.group(2))

    # Extract bounding boxes
    boxes = re.findall(
        r'Bounding box.*:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
        txt_content
    )

    yolo_lines = []
    for (xmin, ymin, xmax, ymax) in boxes:
        xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

    return yolo_lines


def convert_car_or_cycle_annotation(xml_txt_lines, class_id):
    yolo_lines = []

    for line in xml_txt_lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts
        yolo_lines.append(f"{class_id} {x} {y} {w} {h}")

    return yolo_lines


# ---------------------------------------------------------
# PROCESS EACH DATASET
# ---------------------------------------------------------

all_images = []

for name, info in DATASETS.items():
    dataset_path = info["path"]
    class_id = info["class_id"]

    print(f"\nProcessing dataset: {name}")

    for file in os.listdir(dataset_path):
        if not file.endswith(".txt"):
            continue

        annotation_path = os.path.join(dataset_path, file)

        # Prefix for unique names
        prefix = name.lower()

        # Read annotation file
        with open(annotation_path, "r") as f:
            txt = f.read()

        # -------------------------------------------------
        # Pedestrian dataset format
        # -------------------------------------------------
        if name == "Pedestrian":
            yolo_lines = convert_pedestrian_annotation(txt, class_id)

        # -------------------------------------------------
        # Car / Cycle format (already in YOLO but class 0)
        # -------------------------------------------------
        else:
            yolo_lines = convert_car_or_cycle_annotation(txt.splitlines(), class_id)

        if not yolo_lines:
            continue

        # Save YOLO TXT
        output_label_name = prefix + "_" + file.replace(".txt", ".txt")
        output_label_path = os.path.join(OUTPUT_LABELS, output_label_name)

        # Find image
        img_name = file.replace(".txt", ".png")
        base = file.replace(".txt", "")

        possible_ext = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

        img_path = None
        for ext in possible_ext:
            test_path = os.path.join(dataset_path, base + ext)
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None:
            print(f"❌ Image for {file} not found in {dataset_path}")
            continue
        
        # Store for train/val split
        all_images.append((img_path, output_label_path, yolo_lines))


# ---------------------------------------------------------
# TRAIN / VAL SPLIT
# ---------------------------------------------------------

random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_SPLIT)

train_set = all_images[:split_idx]
val_set = all_images[split_idx:]

def save_set(data_list, out_img_dir, out_lbl_dir):
    for img_path, lbl_path, yolo_lines in data_list:
        # Save label
        with open(os.path.join(out_lbl_dir, os.path.basename(lbl_path)), "w") as f:
            f.write("\n".join(yolo_lines))

        # Save image
        shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))


save_set(train_set, f"{OUTPUT_IMAGES}/train", f"{OUTPUT_LABELS}/train")
save_set(val_set, f"{OUTPUT_IMAGES}/val", f"{OUTPUT_LABELS}/val")

# ---------------------------------------------------------
# CREATE YAML FILE
# ---------------------------------------------------------

yaml_path = f"{BASE_PATH}/dataset.yaml"

with open(yaml_path, "w") as f:
    f.write(f"""
train: {OUTPUT_IMAGES}/train
val: {OUTPUT_IMAGES}/val

nc: {len(CLASSES)}
names: {CLASSES}
""")

print("\n✅ Dataset conversion completed!")
print(f"YAML saved to: {yaml_path}")
print("Ready for YOLO training 🚀")
