import json
import logging
import os
import pickle
import random
import re
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.udop import UdopProcessor, UdopTokenizer

from core.datasets.collate_selfSupervised import DataCollatorForSelfSupervisedTasks


logger = logging.getLogger(__name__)


class MIRIDIH_Dataset(Dataset):
    def __init__(
        self,
        processor: UdopProcessor,
        tokenizer: UdopTokenizer,
        data_args,
    ):
        """Structure of data directory:

        ── data/
            ├── images/
            │   └── image_{idx}.png
            └── json_data/
                └── processed_{idx}.pickle
        """
        assert os.path.isdir(
            data_args.data_dir
        ), f"Data dir {data_args.data_dir} does not exist!"
        logger.info("Loading Dataset")

        self.task = data_args.task_name
        self.unit = data_args.unit

        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0
        self.image_size = data_args.image_size
        self.layout_modeling_masking_ratio = (
            0.75 if not data_args.curriculum else data_args.curri_start_MR
        )

        self.cls_collator = DataCollatorForSelfSupervisedTasks(tokenizer=tokenizer)

        json_dir = os.path.join(data_args.data_dir, "json_data")
        img_dir = os.path.join(data_args.data_dir, "images")

        self.json_file = []
        self.images = []

        # Load valid image and json file
        for json_file in tqdm(os.listdir(json_dir)):
            idx = int(re.findall(r"\d+", json_file)[0])
            json_path = os.path.join(json_dir, json_file)
            image_path = os.path.join(img_dir, f"image_{idx}.png")

            if (os.path.isfile(json_path)) and (os.path.isfile(image_path)):
                self.json_file.append(json_path)
                self.images.append(image_path)

        assert len(self.json_file) == len(
            self.images
        ), f"Number of json files and images are not equal!"
        logger.info(f"There are {len(self.images)} images with annotations")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file_ = self.json_file[index]

        with open(file_, "rb") as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_}")

        # Perform masking and encoding data
        encoding = self.mask_selfSupervised(
            json_data,
            self.images[index],
            self.tokenizer,
        )

        encoding["input_ids"] = encoding["input_ids"][0]
        encoding["bbox"] = encoding["bbox"][0]
        encoding["labels"] = encoding["labels"][0]
        encoding["pixel_values"] = encoding["pixel_values"][0]
        encoding["attention_mask"] = encoding["attention_mask"][0]
        encoding._file_name = self.json_file[index]
        encoding._thumbnail_url = json_data["thumbnail_url"]

        return encoding

    def set_layout_modeling_masking_ratio(self, new_ratio):
        self.layout_modeling_masking_ratio = new_ratio

    def get_layout_modeling_masking_ratio(self):
        return self.layout_modeling_masking_ratio

    def mask_selfSupervised(
        self,
        json_data: dict,
        image_dir: str,
        tokenizer: UdopTokenizer,
    ) -> dict:
        image = Image.open(image_dir).convert("RGB")
        width, height = image.size

        task = None
        tokenize_unit = None

        # Assign self-supervised task
        if self.task == "All":
            r = random.randint(0, 2)
            if r == 0:
                task = "Layout Modeling"
            elif r == 1:
                task = "Visual Text Recognition"
            else:
                task = "Joint Text-Layout Reconstruction"
        else:
            task = self.task

        # Assign mask ratio
        if "Layout Modeling" in task:
            mask_ratio = self.layout_modeling_masking_ratio
        elif "Visual Text Recognition" in task:
            mask_ratio = 0.5
        elif "Joint Text-Layout Reconstruction" in task:
            mask_ratio = 0.15

        # Assign tokenize unit
        if self.unit == "word":
            tokenize_unit = "word"
        elif self.unit == "token":
            tokenize_unit = "token"
        else:
            tokenize_unit = self.unit

        assert (
            tokenize_unit == "word" or tokenize_unit == "token"
        ), f"Wrong tokenize unit!"

        # Perform masking
        total_IDs, total_bbox, total_labels = [], [], []

        sentinel_idx = 0
        token_idx = 0

        # Iterate over sentences
        for text in json_data["form"]:
            valid_text = True
            sentence_text, sentence_bbox = [], []
            for word in text["words"]:
                word_text = []
                word_bbox = []
                if word["text"].isspace():
                    continue

                # Normalize bbox to 0 ~ 500
                bbox = [
                    int(500 * (word["box"][0] / width)),
                    int(500 * (word["box"][1] / height)),
                    int(500 * (word["box"][2] / width)),
                    int(500 * (word["box"][3] / height)),
                ]

                # If bounding box is out of range, skip
                valid_text = all(0 <= x <= 500 for x in bbox)
                if not valid_text:
                    break

                if tokenize_unit == "word":
                    sentence_text.append(word["text"])
                    sentence_bbox.append(bbox)
                elif tokenize_unit == "token":
                    sentence_text.extend(word_text)
                    sentence_bbox.extend(word_bbox)

            if not valid_text:
                continue

            assert len(sentence_text) == len(
                sentence_bbox
            ), f"text bbox length mismatch"

            # Determine which tokens to mask and their corresponding bounding boxes
            group_list, group_bbox_list = mask_process(
                sentence_bbox, mask_ratio=mask_ratio
            )

            numbering_list = [
                i for i in range(sentinel_idx, sentinel_idx + len(group_list))
            ]
            sentinel_idx = sentinel_idx + len(group_list)

            if sentinel_idx > 100:  # Mask until sentinel token 99
                break

            for _word in sentence_text:
                token_idx += len(tokenizer.tokenize(_word))
            if token_idx > 510:
                break

            word_list = []
            if tokenize_unit == "word":
                for word_text in sentence_text:
                    word_list.append(word_text)
            elif tokenize_unit == "token":
                word_list = tokenizer.convert_tokens_to_ids(sentence_text)

            # Perform Self-Supervised task
            input_ids, labels, bbox_list = self.cls_collator(
                task,
                word_list,
                sentence_bbox,
                group_list,
                group_bbox_list,
                numbering_list,
            )

            total_IDs.extend(input_ids)
            total_bbox.extend(bbox_list)
            total_labels.extend(labels)

        # Encode data through UDOP Processor
        encoding = self.processor(
            images=image,
            text=[task + "."],
            text_pair=[total_IDs],
            boxes=[total_bbox],
            return_tensors="pt",
        )
        encoding["labels"] = self.processor(
            images=image, text_target=total_labels, return_tensors="pt"
        )["input_ids"]

        encoding["labels"] = encoding["labels"][:, 0].unsqueeze(0)
        encoding["labels"] = torch.cat(
            (encoding["labels"], torch.tensor([tokenizer.eos_token_id]).unsqueeze(0)),
            dim=1,
        )

        return encoding


def mask_process(bbox_list, mask_ratio=0.75) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Main masking process for self-supervised tasks.
    Args:
        bbox_list (`List[List[int, int, int, int]]`): List of bounding boxes
        mask_ratio (`float`): Self-supervised masking ratio

    Returns:
        grouped_tokens (`List[List[int, int]]`): List of slices of continuous tokens
        grouped_bbox (`List[List[int, int, int, int]]`): List of grouped bounding boxes
    """
    l = len(bbox_list)
    mask = random_masking(L=l, mask_ratio=mask_ratio)
    grouped_tokens = group_tokens(mask)
    return grouped_tokens, group_bbox(bbox_list, grouped_tokens)


def random_masking(L=4096, mask_ratio=0.75) -> torch.Tensor:
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


def group_tokens(mask: torch.Tensor) -> List[List[int]]:
    """
    Group masked tokens into chunks of continuous tokens for bounding box grouping.

    Args:
        mask (`List[int]`): binary mask for masked tokens

    Returns:
        group_lst (`List[List[int, int]]`): List of slices of continuous tokens

    """
    group_lst = []
    i = 0
    prev = 0

    for m in mask:
        if m == 0:
            if i == prev:
                pass
            else:
                group_lst.append([prev, i])
            prev = i + 1
        i += 1

    if prev != i:
        group_lst.append([prev, i])

    return group_lst


def group_bbox(bbox_lst, group_lst) -> List[List[int]]:
    """
    Group bounding boxes of masked tokens into chunks of continuous tokens.
    Args:
        bbox_lst (`List[List[int, int, int, int]]`): List of bounding boxes
        group_lst (`List[List[int, int]]`): List of slices of continuous tokens

    Returns:
        bbox_group_lst (`List[List[int, int, int, int]]`): List of grouped bounding boxes

    """
    bbox_group_lst = []

    for s in group_lst:
        target = bbox_lst[s[0] : s[1]]
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
            bbox_group_lst.append([l, t, r, b])

    return bbox_group_lst
