# Data Collator and Masking for Self-Supervised Tasks

class DataCollatorForSelfSupervisedTasks:

    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        
        self.tokenizer = tokenizer
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

        if 'Layout Modeling' in task:
            return self.LM(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
        elif 'Visual Text Recognition' in task:
            return self.VT(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
        elif 'Joint Text-Layout Reconstruction' in task:
            return self.JR(ids_list, sentence_bbox, group_list, group_bbox_list, numbering_list)
        
        else:
            raise ValueError("Invalid user prompt")


class DataCollatorForT5LayoutModeling:
    """
    Data collator used for T5 Layout Modeling
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):
        
        res_input_ids = []
        res_bbox_list = []
        labels = []
        
        for idx in range(len(label_numbering)):
            labels.append(f'<extra_l_id_{label_numbering[idx]}>')
            labels.append(f'<loc_{int(group_bbox_list[idx][0])}>')
            labels.append(f'<loc_{int(group_bbox_list[idx][1])}>')
            labels.append(f'<loc_{int(group_bbox_list[idx][2])}>')
            labels.append(f'<loc_{int(group_bbox_list[idx][3])}>')
            
        slice_pointer=0
        L = len(group_list)
        input_len = len(input_ids)
        mask_flag = False
        for i in range(input_len):
            #input_ids[i] = input_ids[i] if isinstance(input_ids[i], list) else [input_ids[i]]
            if slice_pointer < L and i == group_list[slice_pointer][0]:
                mask_flag = True
                res_input_ids.append(f'<extra_l_id_{label_numbering[slice_pointer]}>')
                res_bbox_list.append([0,0,0,0])
                res_input_ids.append(input_ids[i])
                res_bbox_list.append([0,0,0,0])
            elif slice_pointer < L and i == group_list[slice_pointer][1]:
                mask_flag = False
                res_input_ids.append(f'</extra_l_id_{label_numbering[slice_pointer]}>')
                res_bbox_list.append([0,0,0,0])
                res_input_ids.append(input_ids[i])
                res_bbox_list.append(bbox_list[i])
                slice_pointer += 1
            else:
                if mask_flag:
                    res_bbox_list.append([0,0,0,0])
                else:
                    res_bbox_list.append(bbox_list[i])
                res_input_ids.append(input_ids[i])
                
        if slice_pointer < L and input_len == group_list[slice_pointer][1] :
            res_input_ids.append(f'</extra_l_id_{label_numbering[slice_pointer]}>')
            res_bbox_list.append([0,0,0,0])
        
        return res_input_ids, labels, res_bbox_list

class DataCollatorForT5VisTextRec:
    """
    Data collator used for T5 Visual Text Recognition
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer 
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):

        res_input_ids = []
        res_bbox_list = []
        labels = []

        for idx in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_t_id_{label_numbering[idx]}>', add_special_tokens=False)
            for ids in input_ids[group_list[idx][0]:group_list[idx][1]] :
                ids = ids if isinstance(ids, list) else [ids]
                labels.extend([i for i in ids])

        slice_pointer, idx = 0, 0
        L = len(group_list)
        len_ID = len(input_ids)

        while idx < len_ID:
            input_ids[idx] = input_ids[idx] if isinstance(input_ids[idx], list) else [input_ids[idx]]
            if slice_pointer < L and idx == group_list[slice_pointer][0]:
                res_input_ids += self.tokenizer.encode(f'<extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                res_bbox_list.append([0,0,0,0])

                res_input_ids += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[slice_pointer][0])}>', add_special_tokens=False)
                res_input_ids += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[slice_pointer][1])}>', add_special_tokens=False)
                res_input_ids += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[slice_pointer][2])}>', add_special_tokens=False)
                res_input_ids += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids*group_bbox_list[slice_pointer][3])}>', add_special_tokens=False)
                res_bbox_list += [[0,0,0,0]] * 4
                
                res_input_ids += self.tokenizer.encode(f'</extra_t_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                res_bbox_list.append([0,0,0,0])
                idx = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                res_input_ids.extend([ID for ID in input_ids[idx]])
                res_bbox_list.extend([bbox_list[idx]] * len(input_ids[idx]))

            idx += 1

        return res_input_ids, labels, res_bbox_list


class DataCollatorForT5JointReconstruction:
    """
    Data collator used for T5 Joint Text-Layout Reconstruction
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.tokenizer = tokenizer
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, input_ids, bbox_list, group_list, group_bbox_list, label_numbering):
        
        res_input_ids = []
        res_bbox_list = []
        labels = []

        for idx in range(len(label_numbering)):
            labels += self.tokenizer.encode(f'<extra_id_{label_numbering[idx]}>', add_special_tokens=False)
            for ids in input_ids[group_list[idx][0]:group_list[idx][1]] :
                ids = ids if isinstance(ids, list) else [ids]
                labels.extend([i for i in ids])
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][0])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][1])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][2])}>', add_special_tokens=False)
            labels += self.tokenizer.encode(f'<loc_{int(self.tokenizer._loc_extra_ids * group_bbox_list[idx][3])}>', add_special_tokens=False)

        slice_pointer, idx = 0, 0
        L = len(group_list)
        len_ID = len(input_ids)
        
        while idx < len_ID:
            input_ids[idx] = input_ids[idx] if isinstance(input_ids[idx], list) else [input_ids[idx]]
            if slice_pointer < L and idx == group_list[slice_pointer][0]:
                res_input_ids += self.tokenizer.encode(f'<extra_id_{label_numbering[slice_pointer]}>', add_special_tokens=False)
                res_bbox_list.append([0,0,0,0])

                idx = group_list[slice_pointer][1]-1
                slice_pointer += 1
            else:
                res_input_ids.extend([ID for ID in input_ids[idx]])
                res_bbox_list.extend([bbox_list[idx]] * len(input_ids[idx]))
            
            idx += 1

        return res_input_ids, labels, res_bbox_list
    
