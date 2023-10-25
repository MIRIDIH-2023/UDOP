# Util file for finding and deleting empty json files with image
# Requirement: Specify personal image and JSON folder path

import json
import os
import re
import pickle 

# Specify the image folder and the JSON folder
json_folder_path = "/home/work/sy/UDOP/data/json_data"
image_folder_path = "/home/work/sy/UDOP/data/images"


error_json = []

def process_json(json_folder_path):
    count = 0
    json_files = os.listdir(json_folder_path)
    for json_file in json_files:
        path = os.path.join(json_folder_path, json_file)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            data = json.loads(json.dumps(obj, default=str))
            text = data['form']
            if not text:
                print(json_file)
                error_json.append(json_file)
                count += 1
    print("Total error files: " + str(count))

def delete_files(json_folder_path, image_folder_path):
    for json_file in error_json:
        idx = int(re.findall(r'\d+', json_file)[0])

        json_path = os.path.join(json_folder_path, json_file)
        os.remove(json_path)
        img_path = os.path.join(image_folder_path, f"image_{idx}.png")
        print(img_path)
        os.remove(img_path)


# Call the function to remove files
process_json(json_folder_path)
# delete_files(json_folder_path, image_folder_path)
