import os
import hashlib
import re

def get_image_hashes(folder):
    image_hashes = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                image_data = f.read()
                hash_value = hashlib.md5(image_data).hexdigest()
                image_hashes[file_path] = hash_value
    return image_hashes

def find_duplicate_images(folder):
    image_hashes = get_image_hashes(folder)
    duplicate_images = {}

    for image1, hash1 in image_hashes.items():
        for image2, hash2 in image_hashes.items():
            if image1 != image2 and hash1 == hash2:
                duplicate_images[image1] = image2

    return duplicate_images

def print_duplicate_images(duplicate_images):
    if len(duplicate_images) == 0:
        print("No duplicate images found in the folder.")
    else:
        print("Duplicate images found in the folder:")
        for image1, image2 in duplicate_images.items():
            print(f"Image 1: {image1}\nImage 2: {image2}\n")
    print("Total duplicates: ", len(duplicate_images))

def delete_duplicate_images(duplicate_images):
    for image1, image2 in duplicate_images.items():
        idx = int(re.findall(r'\d+', image2)[0])
        json_path = os.path.join(json_folder_path, f"{idx}.pickle")

        if os.path.exists(image2):
            os.remove(image2)
        if os.path.exists(json_path):
            os.remove(json_path)




# Provide the path to the folder containing the images
folder_path = '/home/work/sy/UDOP/data/images'
json_folder_path = "/home/work/sy/UDOP/data/json_data"

# Find and print the duplicate images
duplicate_images = find_duplicate_images(folder_path)
print_duplicate_images(duplicate_images)
delete_duplicate_images(duplicate_images)