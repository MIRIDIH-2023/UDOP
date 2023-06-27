import json
import logging
import os
import random
import pandas as pd


from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForSelfSupervisedTasks
from io import BytesIO
import requests

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

        json_dir = os.path.join(data_args.data_dir, 'json')
        img_dir = os.path.join(data_args.data_dir, 'images')

        csv_path = os.path.join(data_args.data_dir, 'sample.csv')

        self.main_df = pd.read_csv(csv_path) # xml_sample.csv 파일 저장

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

        results = [self.load_file(file_idx, json_dir, img_dir) for file_idx in tqdm(range(len(os.listdir(json_dir))))]
        for json_file, images in results:
            self.json_file += json_file
            self.images += images
        # assert len(self.labels) == len(self.json_file)
        # logger.info(f'There are {len(self.labels)} images with annotations')

    def load_file(self, file_idx, json_dir, img_dir):
        json_file, images = [], []

        json_path = os.path.join(json_dir, f"processed_sample_{file_idx}.json")
        image_path = os.path.join(img_dir, f"image_{file_idx}.png")

        json_file.append(json_path)
        images.append(image_path)

        return json_file, images
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index): #완료
        # try:
            print("Dataloader:" + str(index))
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
            assert len(bbox_input) == len(input_ids)
            assert len(bbox_input.size()) == 2
            assert len(char_bbox_input.size()) == 2

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
        # except: #오류가 났다는 거는 파일이 없다는 것. 해당 상황에서는 index+1 파일 불러오는 것으로 대체
        #     print(f"{index} 파일을 {index+1}로 대체")
        #     return self.__getitem__(index+1)

            #return self[(index + 1) % len(self)]

    #def get_labels(self): # classification에서 label의 종류 출력하는 함수. 우리는 필요 없을 듯.
    #    return list(map(str, list(range(self.NUM_LABELS))))

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

    # 해당 부분은 json파일의 문장을 line-by-line으로 읽는 것으로 해당 함수 수정 완료.
    def read_ocr_core_engine(self, file_, image_dir, tokenizer, max_seq_length=None, num_img_embeds=None, image_size=224):
        #max_seq_length와 num_img_embeds 는 원본 코드에서도 안쓰는데 왜있는거지?

        with open(file_, 'r', encoding='utf8') as f:
            try:
                data = json.load(f)
            except:
                print(f"wrong in file {file_}")
                data = {}
        rets = []
        n_split = 0


        image =  Image.open(image_dir)
        width, height = image.size
        image = img_trans_torchvision(image, image_size)

        total_text, total_bbox = [], []
        
        for text in data['form']: #문장별로 쪼갬
            sentence_text, sentence_bbox = [], []
            for word in text['words']: #단어별로 쪼갬

                if word == ' ': #띄어쓰기는 건너뛰기
                    continue

                bbox = [ 
                    word['box'][0] / width,
                    word['box'][1] / height,
                    word['box'][2] / width,
                    word['box'][3] / height
                ]

                sub_tokens = tokenizer.tokenize(word['text']) #단어별로 쪼갠걸 다시 토큰화 (하나의 단어도 여러개의 토큰 가능)
                for sub_token in sub_tokens:
                    sentence_text.append(sub_token)
                    sentence_bbox.append(bbox) #현재는 단어별 bbox, 추후 문장별 bbox로도 수정 가능
                    #bbox_list.append(form['box'])
            total_text.append(sentence_text)
            total_bbox.append(sentence_bbox)
        input_ids, labels, bbox_input = self.cls_collator(self.task, total_text, total_bbox) #prompt 붙여서 최종 input,bbox,label을 만듦. ################################


        return input_ids, labels, bbox_input, image
