# # import os
# # import xml.etree.ElementTree as ET

# # XML_FILE = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Car/annotations.xml"
# # OUTPUT_DIR = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Car/"
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # Classes in your dataset
# # classes = ["car", "minivan"]

# # tree = ET.parse(XML_FILE)
# # root = tree.getroot()

# # # Get image size from <meta> section
# # meta = root.find("meta").find("task").find("original_size")
# # IMG_W = int(meta.find("width").text)
# # IMG_H = int(meta.find("height").text)


# # # Dictionary: frame → list of YOLO lines
# # frames = {}

# # # Iterate over tracks
# # for track in root.findall("track"):
# #     label = track.get("label")
# #     cls_id = classes.index(label)

# #     for box in track.findall("box"):
# #         frame_id = int(box.get("frame"))
# #         outside = int(box.get("outside"))

# #         # Skip if object is outside frame
# #         if outside == 1:
# #             continue

# #         xtl = float(box.get("xtl"))
# #         ytl = float(box.get("ytl"))
# #         xbr = float(box.get("xbr"))
# #         ybr = float(box.get("ybr"))

# #         # Convert to YOLO format
# #         x_center = ((xtl + xbr) / 2) / IMG_W
# #         y_center = ((ytl + ybr) / 2) / IMG_H
# #         w = (xbr - xtl) / IMG_W
# #         h = (ybr - ytl) / IMG_H

# #         line = f"{cls_id} {x_center} {y_center} {w} {h}"

# #         if frame_id not in frames:
# #             frames[frame_id] = []
# #         frames[frame_id].append(line)

# # # Write YOLO TXT files
# # for frame_id, yolo_lines in frames.items():
# #     if frame_id >= 300:
# #         break
# #     txt_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:06d}.txt")
# #     with open(txt_path, "w") as f:
# #         f.write("\n".join(yolo_lines))

# # print("Conversion complete! YOLO labels written to /labels/")


# # import os
# # import re

# # # Folder containing multiple Penn-Fudan text annotation files
# # ANNOT_DIR = "/home/ahzsyed/Documents/GitHub/Perception_AutonomousSystems/data/YOLODataset/Pedestrians"

# # # Folder where YOLO labels will be saved
# # OUTPUT_DIR = os.path.join(ANNOT_DIR, "labels")
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # class_id = 0  # "person" for YOLO

# # for filename in os.listdir(ANNOT_DIR):
# #     if not filename.endswith(".txt"):
# #         continue

# #     input_path = os.path.join(ANNOT_DIR, filename)

# #     with open(input_path, "r") as f:
# #         data = f.read()

# #     # Extract image size
# #     size_match = re.search(r'Image size.*:\s*(\d+)\s*x\s*(\d+)', data)
# #     if not size_match:
# #         print(f"Skipping {filename} — no image size found")
# #         continue

# #     img_w = int(size_match.group(1))
# #     img_h = int(size_match.group(2))

# #     # Extract all bounding boxes
# #     boxes = re.findall(
# #         r'Bounding box.*:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)',
# #         data
# #     )

# #     if not boxes:
# #         print(f"No bounding boxes found in {filename}")
# #         continue

# #     yolo_lines = []

# #     for (xmin, ymin, xmax, ymax) in boxes:
# #         xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))

# #         # YOLO conversion
# #         x_center = (xmin + xmax) / 2 / img_w
# #         y_center = (ymin + ymax) / 2 / img_h
# #         w = (xmax - xmin) / img_w
# #         h = (ymax - ymin) / img_h

# #         yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

# #     # Save YOLO file
# #     output_path = os.path.join(
# #         OUTPUT_DIR,
# #         filename.replace(".txt", "_yolo.txt")
# #     )

# #     with open(output_path, "w") as f:
# #         f.write("\n".join(yolo_lines))

# #     print(f"Converted: {filename} → labels/{os.path.basename(output_path)}")

# # print("\nConversion completed for all files!")



import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your dataset
dataset_path = "/zhome/e5/7/219270/Perception_AutonomousSystems/data/YOLODataset"
dataset_path = os.path.expanduser(dataset_path)

# Mapping folder names to YOLO labels
label_map = {
    "Pedestrians": 0,
    "Cycle": 1,
    "Car": 2
}

# Create output directories
output_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
for d in output_dirs:
    os.makedirs(os.path.join(dataset_path, d), exist_ok=True)

# Collect all image files and their corresponding labels
image_files = []
label_files = []

for folder, label in label_map.items():
    folder_path = os.path.join(dataset_path, folder)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    files_in_folder = os.listdir(folder_path)
    print(f"Processing folder '{folder}', {len(files_in_folder)} files found")
    
    for file in files_in_folder:
        if file.lower().endswith((".jpg", ".png")):  # case-insensitive check
            img_path = os.path.join(folder_path, file)
            txt_file = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(folder_path, txt_file)
            
            if os.path.exists(txt_path):
                image_files.append(img_path)
                label_files.append(txt_path)
                
                # Update label inside txt file
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                with open(txt_path, "w") as f:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = str(label)
                            f.write(" ".join(parts) + "\n")
            else:
                print(f"Warning: Annotation missing for {file}")

# Split into train and validation
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)

# Function to copy files
def copy_files(files, labels, subset):
    for img, lbl in zip(files, labels):
        shutil.copy(img, os.path.join(dataset_path, subset, "images"))
        shutil.copy(lbl, os.path.join(dataset_path, subset, "labels"))

# Copy train and val files
copy_files(train_imgs, train_lbls, "train")
copy_files(val_imgs, val_lbls, "val")

print("Dataset split complete!")
print(f"Train: {len(train_imgs)} images")
print(f"Val: {len(val_imgs)} images")







