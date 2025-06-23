# import os
# import re
# from datetime import datetime, timedelta

# # Function to increment the timestamp by 3 hours
# def adjust_timestamp(filename):
#     # Match the timestamp pattern in the filename
#     match = re.search(r'_(\d{4})_LST$', filename)
#     if match:
#         original_time = match.group(1)
#         # Parse the timestamp
#         time_obj = datetime.strptime(original_time, "%H%M")
#         # Add 3 hours
#         adjusted_time = (time_obj + timedelta(hours=3)).strftime("%H%M")
#         # Replace the old timestamp with the adjusted one
#         return filename.replace(f"_{original_time}_LST", f"_{adjusted_time}_LST")
#     return filename

# dry_run = True  # Set to True for testing, False for actual renaming

# def rename_files(base_folder, origin_prefix, log_file, dry_run=False):
#     with open(log_file, "w") as log:
#         for root, _, files in os.walk(base_folder):
#             for file in files:
#                 if file.startswith(origin_prefix):
#                     new_name = adjust_timestamp(file)
#                     if new_name != file:
#                         old_path = os.path.join(root, file)
#                         new_path = os.path.join(root, new_name)
#                         if not dry_run:
#                             os.rename(old_path, new_path)
#                         log.write(f"{old_path} -> {new_path}\n")
#                         print(f"{'Would rename:' if dry_run else 'Renamed:'} {old_path} -> {new_path}")

# # # Function to rename files in all subfolders and log changes
# # def rename_files(base_folder, origin_prefix, log_file):
# #     with open(log_file, "w") as log:
# #         for root, _, files in os.walk(base_folder):
# #             for file in files:
# #                 if file.startswith(origin_prefix):
# #                     new_name = adjust_timestamp(file)
# #                     if new_name != file:
# #                         old_path = os.path.join(root, file)
# #                         new_path = os.path.join(root, new_name)
# #                         os.rename(old_path, new_path)
# #                         # Log the changes
# #                         log.write(f"{old_path} -> {new_path}\n")
# #                         print(f"Renamed: {old_path} -> {new_path}")

# # Specify the base folder, origin prefix, and log file name
# base_folder = "../results/output/1251066/"
# origin_prefix = "1251066"
# log_file = "rename_log.txt"

# # Run the renaming process
# rename_files(base_folder, origin_prefix, log_file)

# print(f"Renaming complete. Changes logged in '{log_file}'.")


import os
import re
from datetime import datetime, timedelta

# Function to increment the timestamp by 3 hours
def adjust_timestamp(filename):
    # Match the timestamp pattern in the filename
    match = re.search(r'_(\d{4})_LST.tif$', filename)
    if match:
        original_time = match.group(1)
        # Parse the timestamp
        time_obj = datetime.strptime(original_time, "%H%M")
        # Add 3 hours
        adjusted_time = (time_obj + timedelta(hours=3)).strftime("%H%M")
        # Replace the old timestamp with the adjusted one
        return filename.replace(f"_{original_time}_LST", f"_{adjusted_time}_LST")
    return filename

# Function to rename files in all subfolders and log changes
def rename_files(base_folder, origin_prefix, log_file, dry_run=False):
    with open(log_file, "w") as log:
        for root, _, files in os.walk(base_folder):
            for file in files:
                print(file)
                # Check if the file starts with the specified origin prefix
                if file.startswith(origin_prefix):
                    new_name = adjust_timestamp(file)
                    if new_name != file:  # Only rename if the name changes
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, new_name)
                        if not dry_run:
                            os.rename(old_path, new_path)
                        log.write(f"{old_path} -> {new_path}\n")
                        print(f"{'Would rename:' if dry_run else 'Renamed:'} {old_path} -> {new_path}")

# Specify the base folder, origin prefix, and log file name
base_folder = "../results/output/1251066/"
origin_prefix = "1251066"
log_file = "rename_log.txt"
dry_run = False  # Set to True to test without renaming

# Run the renaming process
rename_files(base_folder, origin_prefix, log_file, dry_run)

print(f"Renaming complete. Changes logged in '{log_file}'.")