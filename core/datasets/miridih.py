import json
import logging
import os
import random
import re
from io import BytesIO
import pickle

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from core.common.utils import get_visual_bbox, img_trans_torchvision
from core.datasets.collate_supervised import DataCollatorForSelfSupervisedTasks

EMPTY_BOX = [0, 0, 0, 0]
SEP_BOX = [1000, 1000, 1000, 1000]

logger = logging.getLogger(__name__)

class MIRIDIH_Dataset(Dataset):

    def __init__(self , tokenizer , data_args):

        """ Structure of data directory:

            --- sample (.csv)
                   ├── images_url
                   └── labels_url
            --- data (folder)
                   └── processed_sample{index} .json
        """
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info('Loading Dataset')

        json_dir = os.path.join(data_args.data_dir, 'json_data')
        img_dir = os.path.join(data_args.data_dir, 'images')

        # csv_path = os.path.join(data_args.data_dir, 'sample.csv')

        # self.main_df = pd.read_csv(csv_path) # xml_sample.csv 파일 저장

        self.sheet_url = 'sheet_url'
        self.image_url = 'thumbnail_url'
        self.task = data_args.task_name

        self.cls_bbox = EMPTY_BOX[:]
        self.pad_bbox = EMPTY_BOX[:]
        self.sep_bbox = SEP_BOX[:]

        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0

        self.image_size = data_args.image_size

        self.json_file = []
        self.labels = []
        self.images = []

        self.cls_collator = DataCollatorForSelfSupervisedTasks( #기존에 정의한 토크나이저 선언
                tokenizer=tokenizer,
            )

        for json_file in tqdm(os.listdir(json_dir)):
            idx = int(re.findall(r'\d+', json_file)[0])
            json_path = os.path.join(json_dir, json_file)
            image_path = os.path.join(img_dir, f"image_{idx}.png")

            if (os.path.isfile(json_path)) and (os.path.isfile(image_path)):
                self.json_file.append(json_path)
                self.images.append(image_path)
        

        assert len(self.json_file) == len(self.images)
        logger.info(f'There are {self.images} images with annotations')

    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        # print(f"Dataloader on: {self.json_file[index]}, index: {index}")
        input_ids, labels, bbox_input, image = self.read_ocr_core_engine(self.json_file[index], self.images[index] , self.tokenizer, self.max_seq_length, self.num_img_embeds, self.image_size)
        visual_bbox_input = get_visual_bbox(self.image_size) # (x_min, y_min, x_max, y_max) 형태의 좌표로 이루어진 텐서 반환
        attention_mask = [1] * len(input_ids)
        decoder_attention_mask = [1] * len(labels)

        char_list = [0]
        char_bbox_list = [[0,0,0,0]]
        char_ids = torch.tensor(char_list, dtype=torch.long)
        char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)

        bbox_input = torch.tensor(bbox_input, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
        assert len(bbox_input) == len(input_ids), f"BBOX_INPUT != INPUT_IDS on file {self.json_file[index]}, index: {index}"
        assert len(bbox_input.size()) == 2, f"BBOX_INPUT SIZE error on file {self.json_file[index]}, index: {index}"
        assert len(char_bbox_input.size()) == 2, f"char_bbox_input size error on file {self.json_file[index]}, index: {index}"

        return_dict =  {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seg_data": bbox_input,
            "visual_seg_data": visual_bbox_input,
            "decoder_attention_mask": decoder_attention_mask,
            "image": image,
            'char_ids': char_ids,
            'char_seg_data': char_bbox_input
        }
        assert input_ids is not None

        return return_dict

    def pad_tokens(self, input_ids, bbox): #이건 그냥 길이 max_len에 맞게 맞추는 함수
        # [CLS], sentence, [SEP]
        tokenized_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        start_token, _, end_token = tokenized_tokens[0], tokenized_tokens[1:-1], tokenized_tokens[-1]

        sentence = tokenized_tokens
        expected_seq_length = self.max_seq_length - self.num_img_embeds
        mask = torch.zeros(expected_seq_length)
        mask[:len(sentence)] = 1

        bbox = [self.cls_bbox] + bbox + [self.sep_bbox]
        while len(sentence) < expected_seq_length:
            sentence.append(self.tokenizer.pad_token_id)
            bbox.append(self.pad_bbox)

        assert len(sentence) == len(bbox)
        return (sentence, mask, bbox, start_token, end_token)


    def read_ocr_core_engine(self, file_, image_dir, tokenizer, max_seq_length=None, num_img_embeds=None, image_size=224):

        with open(file_, 'rb') as f:
            try:
                obj = pickle.load(f)
                data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_}")

        image =  Image.open(image_dir)
        width, height = image.size
        image = img_trans_torchvision(image, image_size)

        task = None

        if self.task == 'All':
            r = random.randint(0,2)
            if r == 0:
                task = 'Layout Modeling.'
            elif r == 1:
                task = 'Visual Text Recognition.'
            else:
                task = 'Joint Text-Layout Reconstruction.'
        else:
            task = self.task

        if 'Layout Modeling' in task:
            mask_ratio = 0.75
        elif 'Visual Text Recognition' in task:
            mask_ratio = 0.5
        elif 'Joint Text-Layout Reconstruction' in task:
            mask_ratio = 0.15

        
        total_IDs, total_bbox, total_labels = [], [], []

        total_IDs.extend(tokenizer.encode(task, add_special_tokens=False))
        total_bbox += [[0,0,0,0]] * len(total_IDs)

        sentinel_idx = 0
        
        for idx, text in enumerate(data['form']): 
            sentence_text, sentence_bbox = [], []
            for word in text['words']: 

                if word['text'] == ' ':
                    continue

                bbox = [ 
                    word['box'][0] / width,
                    word['box'][1] / height,
                    word['box'][2] / width,
                    word['box'][3] / height
                ]

                sub_tokens = tokenizer.tokenize(word['text']) 
                for sub_token in sub_tokens:
                    sentence_text.append(sub_token)
                    sentence_bbox.append(bbox)

            assert len(sentence_text) == len(sentence_bbox), f"text bbox length mismatch"

            group_list, group_bbox_list = mask_process(sentence_bbox, mask_ratio=mask_ratio)

            numbering_list = [i for i in range(sentinel_idx,sentinel_idx + len(group_list))]
            sentinel_idx = sentinel_idx + len(group_list)

            ids_list = tokenizer.convert_tokens_to_ids(sentence_text)
 
            input_ids, labels, bbox_list = self.cls_collator(task, ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)

            total_IDs.extend(input_ids)
            total_bbox.extend(bbox_list)
            total_labels.extend(labels)
                
            total_IDs.append(tokenizer.eos_token_id)
            total_bbox += [[0,0,0,0]]
            total_labels.append(tokenizer.eos_token_id)

        return total_IDs, total_labels, total_bbox, image


# Argument : ori_bbox_list, mask_ratio
# Returns : token slices to be masked, grouped bboxes
def mask_process(bbox_list, mask_ratio=0.75):
    l = len(bbox_list)
    mask = random_masking(L=l, mask_ratio=mask_ratio)
    grouped_tokens = group_tokens(mask)
    return grouped_tokens, group_bbox(bbox_list, grouped_tokens)

def random_masking(L=4096, mask_ratio=0.75):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(L)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=0)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([L])
    mask[:len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=0, index=ids_restore)
    return mask

# Argument : random_masking의 mask
# Returns : masking할 곳의 slice들
def group_tokens(mask):

    group_lst = []
    i=0
    prev=0

    for m in mask:
        if m == 0:
            if i == prev:
                pass
            else:
                group_lst.append([prev, i])
            prev = i+1
        i += 1

    if prev != i:
        group_lst.append([prev, i])

    return group_lst

# Argument : ori_bbox_lst, group_tokens의 리턴 list (slices)
# Returns : masking된 부분의 그룹화된 bbox
def group_bbox(bbox_lst, group_lst):

    bbox_group_lst = []

    for s in group_lst:
        target = bbox_lst[s[0]:s[1]]
        if len(target) == 1:
            bbox_group_lst.append(*target)
        else:
            t = target[0][0]
            l = target[0][1]
            b = target[0][2]
            r = target[0][3]
            for i in target[1:]:
                if i[0] < t:
                    t = i[0]
                if i[1] < l:
                    l = i[1]
                if i[2] > b:
                    b = i[2]
                if i[3] > r:
                    b = i[3]
            bbox_group_lst.append([t,l,b,r])
    
    return bbox_group_lst