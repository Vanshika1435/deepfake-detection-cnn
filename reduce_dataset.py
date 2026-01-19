import os
import random
import shutil

# Path to dataset folders
DATASET_PATH = "dataset"
NUM_IMAGES_TO_KEEP = 500  # number of images to keep per class

for category in ["real", "fake"]:
    folder_path = os.path.join(DATASET_PATH, category)
    all_images = os.listdir(folder_path)
    
    # If more than NUM_IMAGES_TO_KEEP, randomly delete the rest
    if len(all_images) > NUM_IMAGES_TO_KEEP:
        images_to_delete = random.sample(all_images, len(all_images) - NUM_IMAGES_TO_KEEP)
        for img in images_to_delete:
            img_path = os.path.join(folder_path, img)
            os.remove(img_path)
        print(f"{category}: Deleted {len(images_to_delete)} images, kept {NUM_IMAGES_TO_KEEP}")
    else:
        print(f"{category}: Already {len(all_images)} images, nothing deleted")
