# fix_structure.py - CHẠY 1 LẦN DUY THỨC
import os
import shutil
from pathlib import Path
import random

# ĐƯỜNG DẪN CỦA BẠN (sửa lại cho đúng)
current_dir = Path("train")  # ← thư mục hiện tại chứa lẫn lộn
dataset_dir = Path("smoking_dataset")
train_dir = dataset_dir / "train"
val_dir = dataset_dir / "val"

# Tạo thư mục
for p in [train_dir / "images", train_dir / "labels", val_dir / "images", val_dir / "labels"]:
    p.mkdir(parents=True, exist_ok=True)

# Lấy tất cả file jpg/png
image_files = [f for f in current_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# Ngẫu nhiên chia 90% train, 10% val
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.9)

train_files = image_files[:split_idx]
val_files = image_files[split_idx:]


def copy_files(file_list, dest_img_dir, dest_lbl_dir):
    for img_file in file_list:
        # Copy image
        shutil.copy(img_file, dest_img_dir / img_file.name)

        # Copy label (nếu có)
        label_file = current_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy(label_file, dest_lbl_dir / label_file.name)
        else:
            print(f"Không tìm thấy label: {label_file.name}")


# Copy train
copy_files(train_files, train_dir / "images", train_dir / "labels")

# Copy val
copy_files(val_files, val_dir / "images", val_dir / "labels")

print(f"HOÀN TẤT!")
print(f"→ Train: {len(train_files)} ảnh")
print(f"→ Val: {len(val_files)} ảnh")
print(f"→ Dataset đã sẵn sàng tại: {dataset_dir}")