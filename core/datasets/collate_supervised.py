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


    def __call__(self, task, ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list):

        if task == 'Layout Modeling':
            return self.LM(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
        elif task == 'Visual Text Recognition':
            return self.VT(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
        elif task == 'Joint Text-Layout Reconstruction':
            return self.JR(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
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

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        
        res_input_ids = []
        res_bbox_list = []

        labels = []
        for idx in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_l_id_{label_numbering[idx]}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[idx][0])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[idx][1])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[idx][2])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[idx][3])}>', add_special_tokens=False)
            
        slice_pointer=0
        L = len(group_list)
        input_len = len(input_ids)
        for i in range(input_len):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                res_input_ids += self.tokenizer.encode(f'<extra_l_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                res_input_ids.append(input_ids[i])
                res_bbox_list.append([0,0,0,0])
                res_bbox_list.append(bbox_list[i])
            elif slice_pointer < L and i == group_list[slice_pointer][1] :
                res_input_ids += self.tokenizer.encode(f'</extra_l_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                res_input_ids.append(input_ids[i])
                res_bbox_list.append([0,0,0,0])
                res_bbox_list.append(bbox_list[i])
                slice_pointer += 1
            else:
                res_input_ids.append(input_ids[i])
                res_bbox_list.append(bbox_list[i])
                
        if slice_pointer < L and input_len == group_list[slice_pointer][1] :
            res_input_ids += self.tokenizer.encode(f'</extra_l_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
            res_bbox_list.append([0,0,0,0])
        
        return res_input_ids, labels, res_bbox_list

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

    # Sub Class에서 해야 할 일
    # 1. Argument를 user_prompt, ori_input_ids, group_list, ori_bbox_list, label_numbering을 받는다.
    # 2. label_numbering의 시작이 0이라면, user_prompt를 앞에 id로 변환해서 붙인다. (input_ids에)
    # 3. label_numbering에 따라서 sentinel token을 ori_bbox_list를 보면서 붙인다. (input_ids에)
    # 4. labeling은 group_list와 ori_bbox_list를 보면서 붙인다 (labels에)
    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box
        tmp_input_ids = []
        tmp_bbox_list = []

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        labels = []
        for i in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_t_id_{label_numbering[i]}>', add_special_tokens=True)[:-1]
            labels += input_ids[group_list[i][0]:group_list[i][1]]


        slice_pointer=0
        L = len(group_list)
        for i in range(len(input_ids)):
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                tmp_input_ids += self.tokenizer.encode(f'<extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                #print(f'extra : {len(self.tokenizer.encode(f"<extra_t_id_{label_numbering[slice_pointer]}>", add_special_tokens=True)[:-1])}')
                tmp_bbox_list.append([0,0,0,0])
                bbox_ids = []
                for j in range(4):
                    if j % 2 == 1:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j])}>', add_special_tokens=True)[:-1]
                        #print(f'loc : {len(self.tokenizer.encode(f"<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[1])}>", add_special_tokens=True)[:-1])}')
                    else:
                        bbox_ids += self.tokenizer.encode(f'<loc_{int(500*group_bbox_list[slice_pointer][j])}>', add_special_tokens=True)[:-1]
                        #print(f'loc : {len(self.tokenizer.encode(f"<loc_{int(500*group_bbox_list[slice_pointer][j]/page_size[0])}>", add_special_tokens=True)[:-1])}')
                    tmp_bbox_list.append([0,0,0,0])
                tmp_input_ids += bbox_ids
                tmp_input_ids += self.tokenizer.encode(f'</extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=True)[:-1]
                #print(f'extra : {len(self.tokenizer.encode(f"</extra_t_id_{label_numbering[slice_pointer]}>", add_special_tokens=True)[:-1])}')
                tmp_bbox_list.append([0,0,0,0])
                i = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_bbox_list.append(bbox_list[i])

        return tmp_input_ids, labels, tmp_bbox_list


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

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):
        
        tmp_input_ids = []
        tmp_bbox_list = []

        # TODO: 라벨링 하기 (Visual_Text_Recognition_Test.py 참고하면서)
        labels = []
        for idx in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_id_{label_numbering[idx]}>', add_special_tokens=False)
            labels += input_ids[group_list[idx][0]:group_list[idx][1]]
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][0])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][1])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][2])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][3])}>', add_special_tokens=False)

        slice_pointer, idx = 0, 0
        L = len(group_list)
        len_ID = len(input_ids)
        
        while idx < len_ID:
            if slice_pointer < L and idx == group_list[slice_pointer][0]:
                tmp_input_ids += self.tokenizer.encode(f'<extra_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                tmp_bbox_list.append([0,0,0,0])

                idx = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                tmp_input_ids.append(input_ids[idx])
                tmp_bbox_list.append(bbox_list[idx])
            
            idx += 1
        
        # # TOOD: DEBUG. TO REMOVE
        # for idx in range(len(input_ids)):
        #     print(f"Token {idx}: {self.tokenizer.decode([input_ids[idx]])}", end ='\t')
        # print("\nMask: ", group_list)
        # print("Masked Text: ", self.tokenizer.decode(tmp_input_ids))
        # print("Target Text: ", self.tokenizer.decode(labels))
        # print()

        return tmp_input_ids, labels, tmp_bbox_list
    
