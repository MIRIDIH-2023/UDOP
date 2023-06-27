import math
import collections
import pickle
import os
import random
import torch

import numpy as np
from transformers import PreTrainedTokenizerBase


class DataCollatorForSelfSupervisedTasks:

    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        
        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

        self.LM = DataCollatorForT5LayoutModeling(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.VT = DataCollatorForT5VisTextRec(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )

        self.JR = DataCollatorForT5JointReconstruction(
            tokenizer = self.tokenizer,
            input_length = self.input_length,
            target_length = self.target_length,
            pad_token_id = self.pad_token_id,
            decoder_start_token_id = self.decoder_start_token_id
        )


    def __call__(self, user_prompt, ori_input_ids, ori_bbox_list):

        if 'Layout Modeling' in user_prompt:
            return self.LM(user_prompt, ori_input_ids, ori_bbox_list)
        
        elif 'Visual Text Recognition' in user_prompt:
            return self.VT(user_prompt, ori_input_ids, ori_bbox_list)
        
        elif 'Joint Text-Layout Reconstruction' in user_prompt:
            return self.JR(user_prompt, ori_input_ids, ori_bbox_list)
        
        else:
            raise ValueError("Invalid user prompt")


class DataCollatorForT5LayoutModeling:
    """
    Data collator used for T5 Layout Modeling
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        
        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        # TODO: 라벨링 하기 (Layout_Modeling_Test.py 참고하면서)
        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list


class DataCollatorForT5VisTextRec:
    """
    Data collator used for T5 Visual Text Recognition
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box

        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        if(labels!=None):  #label은 classification에서만 수행
        #인줄 알았는데 layout modeling 이런것도 다 output이 있으니까 label==output 인건가..???
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list


class DataCollatorForT5JointReconstruction:
    """
    Data collator used for T5 Joint Text-Layout Reconstruction
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_text, ori_bbox_list):

        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        masked_input_text, labels, bbox = self.random_masking(ori_input_text, ori_bbox_list, mask_ratio=0.15)
        masked_input_ids = self.tokenizer.encode(masked_input_text, add_special_tokens=True)
        input_ids = prompt_ids + masked_input_ids
        bbox_list = [[0, 0, 0, 0]] * len(prompt_ids) + bbox + [[0, 0, 0, 0]] # <\s> token
        label_ids = self.tokenizer.encode(labels, add_special_tokens=True)

        assert len(input_ids) == len(bbox_list)
        return input_ids, label_ids, bbox_list
        
        
    def random_masking(self, input_text, bbox_list, mask_ratio=0.15):
        idx = 0
        
        total_input_ids, total_targets, total_bbox = [], "", []

        for text,bbox in zip(input_text, bbox_list):
            masked_input_ids, targets, part_bbox = [], "", []

            L = len(text)
            len_keep = int(L * (1 - mask_ratio))

            noise = torch.rand(L)  # Noise in [0, 1]

            # Sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=0)  # Ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=0)

            # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([L])
            mask[:len_keep] = 0
            # Unshuffle to get the binary mask
            mask = torch.gather(mask, dim=0, index=ids_restore)
            mask[0] = 1

            # Convert masked tokens to the '<extra_ids_idx>' format
            is_previous_masked = False
            previous_bbox = None

            for index, (token, token_bbox, is_masked) in enumerate(zip(text, bbox, mask)):
                if is_masked:
                    if not is_previous_masked:
                        masked_input_ids.append(f"<extra_id_{idx}>")
                        tokenized_bbox = list(map(int, [i * self.tokenizer._loc_extra_ids for i in token_bbox]))
                        targets += f" <extra_id_{idx}> {token}" 
                        part_bbox.append([0, 0, 0, 0])
                        idx += 1

                    else: # Previous masked
                        targets += f" {token}"
                        tokenized_bbox = list(map(int, [i * self.tokenizer._loc_extra_ids for i in token_bbox]))
                        tokenized_bbox = [min(previous_bbox[0], tokenized_bbox[0]), min(previous_bbox[1], tokenized_bbox[1]), max(previous_bbox[2], tokenized_bbox[2]), max(previous_bbox[3], tokenized_bbox[3])]

                    previous_bbox = tokenized_bbox
                        
                else: # not masked
                    masked_input_ids.append(token)
                    part_bbox.append(token_bbox)
                    if previous_bbox is not None:
                        targets += f" <loc_{previous_bbox[0]}> <loc_{previous_bbox[1]}> <loc_{previous_bbox[2]}> <loc_{previous_bbox[3]}>"
                        previous_bbox = None
                is_previous_masked = is_masked
            
            if previous_bbox is not None:
                targets += f" <loc_{previous_bbox[0]}> <loc_{previous_bbox[1]}> <loc_{previous_bbox[2]}> <loc_{previous_bbox[3]}>"

            total_input_ids.extend(masked_input_ids)
            total_targets += targets
            total_bbox.extend(part_bbox)

        return total_input_ids, total_targets, total_bbox
    