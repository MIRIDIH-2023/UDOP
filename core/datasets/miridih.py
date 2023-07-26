import json
import logging
import os
import pickle
import random
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from core.common.utils import get_visual_bbox, img_trans_torchvision
from core.datasets.collate_selfSupervised import \
    DataCollatorForSelfSupervisedTasks

logger = logging.getLogger(__name__)

class MIRIDIH_Dataset(Dataset):

    def __init__(self , tokenizer , data_args):

        """ Structure of data directory:

            --- images (folder)
                   └── image_{index}.png
            --- json_data (folder)
                   └── processed_{index}.pickle
        """
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info('Loading Dataset')

        self.task = data_args.task_name
        self.unit = data_args.unit

        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0
        self.image_size = data_args.image_size

        self.cls_collator = DataCollatorForSelfSupervisedTasks(tokenizer=tokenizer)

        json_dir = os.path.join(data_args.data_dir, 'json_data')
        img_dir = os.path.join(data_args.data_dir, 'images')

        self.json_file = []
        self.images = []


        for json_file in tqdm(os.listdir(json_dir)):
            idx = int(re.findall(r'\d+', json_file)[0])
            json_path = os.path.join(json_dir, json_file)
            image_path = os.path.join(img_dir, f"image_{idx}.png")

            if (os.path.isfile(json_path)) and (os.path.isfile(image_path)):
                self.json_file.append(json_path)
                self.images.append(image_path)
        

        assert len(self.json_file) == len(self.images), f"Number of json files and images are not equal!"
        logger.info(f'There are {len(self.images)} images with annotations')

    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        file_ = self.json_file[index]

        with open(file_, 'rb') as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_}")
            
        input_ids, labels, bbox_input, image = self.mask_selfSupervised(json_data, self.images[index] , self.tokenizer, self.max_seq_length, self.num_img_embeds, self.image_size)
        visual_bbox_input = get_visual_bbox(self.image_size) 
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
            'char_seg_data': char_bbox_input,
            "file_name": self.json_file[index],
            "thumbnail_url": json_data['thumbnail_url']
        }
        assert input_ids is not None

        return return_dict


    def mask_selfSupervised(self, json_data, image_dir, tokenizer, max_seq_length=None, num_img_embeds=None, image_size=224):
        image =  Image.open(image_dir)
        width, height = image.size
        image = img_trans_torchvision(image, image_size)

        task = None

        if self.task == 'All':
            r = random.randint(0,2)
            if r == 0:
                task = 'Layout Modeling'
            elif r == 1:
                task = 'Visual Text Recognition'
            else:
                task = 'Joint Text-Layout Reconstruction'
        else:
            task = self.task

        if 'Layout Modeling' in task:
            mask_ratio = 0.75
        elif 'Visual Text Recognition' in task:
            mask_ratio = 0.5
        elif 'Joint Text-Layout Reconstruction' in task:
            mask_ratio = 0.15

        tokenize_unit = None

        if self.unit == 'word' :
            tokenize_unit = 'word'
        elif self.unit == 'token' :
            tokenize_unit = 'token'
        else :
            tokenize_unit = self.unit

        assert (tokenize_unit == 'word' or tokenize_unit == 'token'), f"Wrong tokenize unit!"
        
        total_IDs, total_bbox, total_labels = [], [], []

        total_IDs.extend(tokenizer.encode(task+'.', add_special_tokens=False))
        total_bbox += [[0,0,0,0]] * len(total_IDs)

        sentinel_idx = 0
        
        for text in json_data['form']: 
            valid_text = True
            sentence_text, sentence_bbox = [], []
            for word in text['words']:
                word_text= []
                word_bbox= []
                if word['text'].isspace():
                    continue

                bbox = [ 
                    word['box'][0] / width,
                    word['box'][1] / height,
                    word['box'][2] / width,
                    word['box'][3] / height
                ]

                valid_text = all(0 < x < 1 for x in bbox)
                if not valid_text:
                    break
                
                sub_tokens = tokenizer.tokenize(word['text']) 
                for sub_token in sub_tokens:
                    word_text.append(sub_token)
                    word_bbox.append(bbox)
                
                if tokenize_unit == 'word' :
                    sentence_text.append(word_text)
                    sentence_bbox.append(bbox)
                elif tokenize_unit == 'token' :
                    sentence_text.extend(word_text)
                    sentence_bbox.extend(word_bbox)
            
            if not valid_text:
                continue

            assert len(sentence_text) == len(sentence_bbox), f"text bbox length mismatch"

            group_list, group_bbox_list = mask_process(sentence_bbox, mask_ratio=mask_ratio)

            numbering_list = [i for i in range(sentinel_idx,sentinel_idx + len(group_list))]
            sentinel_idx = sentinel_idx + len(group_list)

            if sentinel_idx > 100:      # Mask until sentinel token 99
                break
            
            ids_list = []
            if tokenize_unit == 'word' :
                for word_text in sentence_text:
                    ids_list.append(tokenizer.convert_tokens_to_ids(word_text))
            elif tokenize_unit == 'token' :
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
            l = target[0][0]
            t = target[0][1]
            r = target[0][2]
            b = target[0][3]
            for i in target[1:]:
                if i[0] < l:
                    l = i[0]
                if i[1] < t:
                    t = i[1]
                if i[2] > r:
                    r = i[2]
                if i[3] > b:
                    b = i[3]
            bbox_group_lst.append([l,t,r,b])
    
    return bbox_group_lst