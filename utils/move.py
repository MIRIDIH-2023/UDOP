import os
import re
import shutil
from tqdm import tqdm

# Define source and destination directories
source_directory = '/home/work/sy/UDOP-SY2/data/images'
json_directory = '/home/work/sy/UDOP-SY2/data/json_data'
destination_directory = '/home/work/sy/UDOP-SY2/data2/images'
dest_json_directory = '/home/work/sy/UDOP-SY2/data2/json_data'

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Define a regular expression pattern to match 'image_{int}' filenames
pattern = re.compile(r'image_(\d+)')

# Function to get the integer value from a filename
def extract_integer(filename):
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    return None

# Loop through the files in the source directory
lst = sorted(os.listdir(source_directory))
lst = lst[671:]
for filename in tqdm(lst):
    source_path = os.path.join(source_directory, filename)

    # Check if the file matches the pattern
    integer_value = extract_integer(filename)
    sourcejson_path = os.path.join(json_directory, f"processed_{integer_value}.pickle")
    
    if integer_value is not None:
        # Increment the integer value by 100000
        new_integer_value = integer_value

        # Create the new filename with the modified integer value
        new_imagename = f'image_{new_integer_value}.png'
        new_jsonname = f'processed_{new_integer_value}.pickle'
        
        # Construct the destination path
        destination_path = os.path.join(destination_directory, new_imagename)
        destinationjson_path = os.path.join(dest_json_directory, new_jsonname)

        # Copy the file to the destination directory with the new filename
        shutil.copyfile(source_path, destination_path)
        shutil.copyfile(sourcejson_path, destinationjson_path)

# Done! Files have been copied to the destination directory with modified integer values.
