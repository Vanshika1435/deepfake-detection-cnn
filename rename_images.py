import os

DATASET_PATH = "dataset"

for category in ["real", "fake"]:
    folder_path = os.path.join(DATASET_PATH, category)
    images = sorted(os.listdir(folder_path))  # sort for consistency
    
    for idx, img_name in enumerate(images, start=1):
        ext = os.path.splitext(img_name)[1]  # keep original extension
        new_name = f"{idx:03d}{ext}"  # 001.jpg, 002.jpg ...
        os.rename(os.path.join(folder_path, img_name), os.path.join(folder_path, new_name))
    
    print(f"{category}: Renamed {len(images)} images")
