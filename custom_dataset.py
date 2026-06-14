import os
import shutil

base = "reid-data"

for split in ["bounding_box_train", "query", "bounding_box_test"]:
    split_path = os.path.join(base, split)

    for person_folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, person_folder)

        # Skip if it's already a file
        if not os.path.isdir(folder_path):
            continue

        # Extract person id
        if "person_" not in person_folder:
            continue

        person_id = int(person_folder.split("_")[1])
        new_id = f"{person_id:04d}"

        for i, img in enumerate(os.listdir(folder_path)):
            src = os.path.join(folder_path, img)
            dst = os.path.join(split_path, f"{new_id}_c1_{i:04d}.jpg")

            shutil.move(src, dst)

        # Remove folder after moving
        

print("Conversion completed ✅")