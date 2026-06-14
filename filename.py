import os

folder_path = "reid-data/market1501/bounding_box_test"   # e.g. "reid-data/market1501/Market-1501-v15.09.15/bounding_box_train"

for filename in os.listdir(folder_path):
    if "_c1_" in filename:
        new_filename = filename.replace("_c1_", "_c2_")
        
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_path, new_path)
        print(f"{filename} → {new_filename}")

print("Renaming completed.")