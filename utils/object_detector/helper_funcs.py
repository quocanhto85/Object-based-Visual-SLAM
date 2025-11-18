# 1. Delete all .png files in the root directory of Google Drive

import os

drive_path = '/content/drive/MyDrive/'

# Get a list of all items in the directory
items_in_drive = os.listdir(drive_path)

# Iterate through the items and delete if it's a file and ends with .png
for item_name in items_in_drive:
    file_path = os.path.join(drive_path, item_name)
    if os.path.isfile(file_path) and file_path.endswith('.png'):
        try:
            os.remove(file_path)
            print(f"Deleted .png file in root directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    elif os.path.isdir(file_path):
        print(f"Skipping: {file_path} (is a directory)")
    else:
        print(f"Skipping: {file_path} (not a .png file in the root directory)")
        
