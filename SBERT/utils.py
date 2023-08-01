from zipfile import ZipFile
import os
import pickle
import torch

def has_five_digits(string):
    count = 0
    for char in string:
        if char.isdigit():
            count += 1
    return count >= 5

def is_valid_keyword(keyword):
  if keyword.isdigit():
    return False
  if has_five_digits(keyword):
    return False
  return True


def unzip_data(data_path, extract_path):
    with ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

"""process data. if data doesn't exist, unzip folder"""
def processing_data(folder_path, data_path=None, extract_path=None):

    if (not os.path.exists(folder_path)):
        unzip_data(data_path=data_path, extract_path=extract_path)

    file_names = os.listdir(folder_path)

    data_list = []
    for file_name in file_names:
        if file_name.startswith("processed_") and file_name.endswith(".pickle"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)
    
    keyword_list = []
    text_list = []
    new_data_list = []

    for i in range(len(data_list)):
        keyword = data_list[i]['keyword']
        keyword = [word for word in keyword if is_valid_keyword(word)]
        if len(keyword) == 0:
            continue
        keyword_list.append(" ".join(keyword))
        text = ""
        for j in range(len(data_list[i]['form'])):
            if type(data_list[i]['form'][j]['text']) == str:
                text += data_list[i]['form'][j]['text'] + "\n"
        text_list.append(text)
        new_data_list.append(data_list[i])

    data_list = new_data_list
    
    return keyword_list, text_list, data_list
