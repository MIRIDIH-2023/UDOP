import logging
import math
import os
import re
import warnings
from io import BytesIO
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)
PREFIX_CHECKPOINT_DIR = 'checkpoint'
_re_checkpoint = re.compile(r'^' + PREFIX_CHECKPOINT_DIR + r'\-(\d+)$')
    
    
def get_visual_bbox(image_size=224):
    image_feature_pool_shape = [image_size//16, image_size//16]
    visual_bbox_x = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[1] + 1),
        1.0,
    ) / image_feature_pool_shape[1])
    visual_bbox_y = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[0] + 1),
        1.0,
    ) / image_feature_pool_shape[0])
    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
            visual_bbox_x[1:].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
        ],
        dim=-1,
    ).view(-1, 4)
    return visual_bbox_input


class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
    
    
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path for path in content if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints,
            key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def img_trans_torch(image, image_size=224):
    trans = T.Compose([
            T.ToTensor(),
            T.Resize([image_size,image_size]),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image


def img_resize(image, image_size=224):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
        ])

    image = trans(image)  # copy to make it writeable
    return image


def img_trans_torchvision(image, image_size=224):

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image

def undo_img_trans_torchvision(image):
    # Undo the transformations applied by img_trans_torchvision
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
    
    image *= std
    image += mean
    image = np.clip(image, 0, 1)
    
    return np.transpose(image, (2, 0, 1)) # Convert back to CHW format


def img_trans_torchvision_int(image, image_size=384):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    image = trans(image)  # copy to make it writeable
    return image

def load_image(image_path):
    image = Image.open(image_path).resize((224,224)).convert('RGB')
    h, w = image.size
    image = torch.tensor(np.array(image))
    return image, (w, h)

def convert_img_to_numpy(img):
    return np.array(img)

def normalize_bbox(bbox, size, scale=1000):
    return [
        int(clamp((scale * bbox[0] / size[0]), 0, scale)),
        int(clamp((scale * bbox[1] / size[1]), 0, scale)),
        int(clamp((scale * bbox[2] / size[0]), 0, scale)),
        int(clamp((scale * bbox[3] / size[1]), 0, scale))
    ]

def random_split(dataset, lengths: Sequence[Union[int, float]],
                 generator = default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# TODO: check for the case when boxes are rotated
def calculate_iou(box1, box2):
    # Parse x1, y1, x2, y2 from location tokens
    x1_1, y1_1, x2_1, y2_1 = [int(box1[i][5:-1]) for i in range(4)]
    x1_2, y1_2, x2_2, y2_2 = [int(box2[i][5:-1]) for i in range(4)]


    # Calculate intersection area
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    # Calculate union area
    box1_area = np.abs(x2_1 - x1_1) * np.abs(y2_1 - y1_1)
    box2_area = np.abs(x2_2 - x1_2) * np.abs(y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def add_bbox_to_image(original_image, tokens, color: Tuple[float, float, float, float] = (1, 1, 1, 1)):
    r, g, b, a = color

    image_np = original_image.clone().permute(1, 2, 0).numpy()

    for token in tokens:
        x1, y1, x2, y2 = token['bbox']

        x1 = round(x1 * original_image.shape[2] / 500)
        x2 = round(x2 * original_image.shape[2] / 500)
        y1 = round(y1 * original_image.shape[1] / 500)
        y2 = round(y2 * original_image.shape[1] / 500)

        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        image_np[y1:y2, x1:x2, 0] = a * (image_np[y1:y2, x1:x2, 0] + r)
        image_np[y1:y2, x1:x2, 1] = a * (image_np[y1:y2, x1:x2, 1] + g)
        image_np[y1:y2, x1:x2, 2] = a * (image_np[y1:y2, x1:x2, 2] + b) 

    new_image = torch.from_numpy(image_np).permute(2, 0, 1)

    return new_image

def parse_token(s):
    pattern_full = r'<extra_id_(\d+)>\s*(.*?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    pattern_without_text = r'<extra_l_id_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    pattern_without_bbox = r'<extra_t_id_(\d+)>(.*?)$'

    if re.match(pattern_full, s):
        matches = re.findall(pattern_full, s)
        tokens = []
        for match in matches:
            id = int(match[0])
            text = match[1]
            bbox = tuple(map(int, match[2:]))
            tokens.append({ 'id': id, 'text': text, 'bbox': bbox })
    elif re.match(pattern_without_text, s):
        matches = re.findall(pattern_without_text, s)
        tokens = []
        for match in matches:
            id = int(match[0])
            bbox = tuple(map(int, match[1:]))
            tokens.append({ 'id': id, 'bbox': bbox })
    elif re.match(pattern_without_bbox, s):
        matches = re.findall(pattern_without_bbox, s)
        tokens = []
        for match in matches:
            id = int(match[0])
            text = match[1]
            tokens.append({ 'id': id, 'text': text })
    else:
        return "Error: The string does not match any known patterns."

    return tokens

def parse_input(s):
    pattern_text = r'<extra_l_id_(\d+)>\s*(.*?)\s*</extra_l_id_\1>'
    pattern_bbox = r'<extra_t_id_(\d+)>\s*<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>\s*</extra_t_id_\1>'

    if re.search(pattern_text, s):
        matches = re.findall(pattern_text, s)
        tokens = []
        for match in matches:
            id = int(match[0])
            text = match[1]
            tokens.append({ 'id': id, 'text': text })
    elif re.search(pattern_bbox, s):
        matches = re.findall(pattern_bbox, s)
        tokens = []
        for match in matches:
            id = int(match[0])
            bbox = tuple(map(int, match[1:]))
            tokens.append({ 'id': id, 'bbox': bbox })
    else:
        return "Error: The string does not match any known patterns."

    return tokens


def visualize_text_layout_task(sample, label_text, prediction_text, data_args, output_dir, index):
    idx_ = int(re.findall(r'\d+', sample['file_name'][0])[0])
    image_path = os.path.join(data_args.data_dir , 'images', f'image_{idx_}.png')
    original_image = T.ToTensor()(Image.open(image_path))
    # original_image = undo_img_trans_torchvision(sample['image'])
    label_tokens = parse_token(label_text)
    prediction_tokens = parse_token(prediction_text)
    masked_image = add_bbox_to_image(original_image, label_tokens, (0, 1, 0, 0.5))
    prediction_masked_image = add_bbox_to_image(original_image, prediction_tokens, (0, 0, 1, 0.5))
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Joint Text-Layout Reconstruction", fontsize=14, fontweight='bold')

    axs[0][0].imshow(original_image.permute(1, 2, 0))
    axs[0][0].set_title('Original Image')

    axs[0][1].imshow(masked_image.permute(1, 2, 0))
    for token in label_tokens:
        x1, y1, x2, y2 = token['bbox']
        x1 = round(x1 * (original_image.shape[2] - 1) / 500)
        x2 = round(x2 * (original_image.shape[2] - 1) / 500)
        y1 = round(y1 * (original_image.shape[1] - 1) / 500)
        y2 = round(y2 * (original_image.shape[1] - 1) / 500)
        axs[0][1].text(x1, y1, token['id'], fontsize=8, bbox=dict(alpha=0.2))
    axs[0][1].set_title('Masked Image (Label Text)')

    axs[0][2].imshow(prediction_masked_image.permute(1, 2, 0))
    for token in prediction_tokens:
        x1, y1, x2, y2 = token['bbox']
        x1 = round(x1 * (original_image.shape[2] - 1) / 500)
        x2 = round(x2 * (original_image.shape[2] - 1) / 500)
        y1 = round(y1 * (original_image.shape[1] - 1) / 500)
        y2 = round(y2 * (original_image.shape[1] - 1) / 500)
        axs[0][2].text(x1, y1, token['id'], fontsize=8, bbox=dict(alpha=0.2))
    axs[0][2].set_title('Prediction Masked Image')

    label_text = '\n'.join([f"{token['id']}: {token['text']}" for token in label_tokens])
    prediction_text = '\n'.join([f"{token['id']}: {token['text']}" for token in prediction_tokens])

    axs[1][0].axis('off') 

    axs[1][1].text(0.5, 1, label_text, fontsize=12, verticalalignment='top', horizontalalignment='center')
    axs[1][1].axis('off')

    axs[1][2].text(0.5, 1, prediction_text, fontsize=12, verticalalignment='top', horizontalalignment='center')
    axs[1][2].axis('off')

    plt.tight_layout()
    plt.show()
    if data_args.do_save_visualize:
        xml = int(re.findall(r'\d+', sample['file_name'][0])[0])
        fig.savefig(os.path.join(output_dir, f'{index}_xml{xml}.png'))

def visualize_text_task(sample, label_text, prediction_text, input_text, data_args, output_dir, index):
    idx_ = int(re.findall(r'\d+', sample['file_name'][0])[0])
    image_path = os.path.join(data_args.data_dir , 'images', f'image_{idx_}.png')
    original_image = T.ToTensor()(Image.open(image_path))
    label_tokens = parse_token(label_text)
    prediction_tokens = parse_token(prediction_text)
    input_tokens = parse_input(input_text)
    masked_image = add_bbox_to_image(original_image, input_tokens, (0, 1, 0, 0.5))

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Visual Text Recognition", fontsize=14, fontweight='bold')

    axs[0][0].imshow(original_image.permute(1, 2, 0))
    axs[0][0].set_title('Original Image')

    axs[0][1].imshow(masked_image.permute(1, 2, 0))
    for input_token, label_token in zip(input_tokens, label_tokens):
        x1, y1, x2, y2 = input_token['bbox']
        x1 = round(x1 * (original_image.shape[2] - 1) / 500)
        x2 = round(x2 * (original_image.shape[2] - 1) / 500)
        y1 = round(y1 * (original_image.shape[1] - 1) / 500)
        y2 = round(y2 * (original_image.shape[1] - 1) / 500)
        axs[0][1].text(x1, y1, label_token['id'], fontsize=8, bbox=dict(alpha=0.2))
    axs[0][1].set_title('Masked Image (Label Text)')

    axs[0][2].imshow(masked_image.permute(1, 2, 0))
    for input_token, prediction_token in zip(input_tokens, prediction_tokens):
        x1, y1, x2, y2 = input_token['bbox']
        x1 = round(x1 * (original_image.shape[2] - 1) / 500)
        x2 = round(x2 * (original_image.shape[2] - 1) / 500)
        y1 = round(y1 * (original_image.shape[1] - 1) / 500)
        y2 = round(y2 * (original_image.shape[1] - 1) / 500)
        axs[0][2].text(x1, y1, prediction_token['id'], fontsize=8, bbox=dict(alpha=0.2))
    axs[0][2].set_title('Masked Image (Prediction Text)')

    label_text = '\n'.join([f"{token['id']}: {token['text']}" for token in label_tokens])
    prediction_text = '\n'.join([f"{token['id']}: {token['text']}" for token in prediction_tokens])

    axs[1][0].axis('off')

    axs[1][1].text(0.5, 1, label_text, fontsize=12, verticalalignment='top', horizontalalignment='center')
    axs[1][1].axis('off')

    axs[1][2].text(0.5, 1, prediction_text, fontsize=12, verticalalignment='top', horizontalalignment='center')
    axs[1][2].axis('off')

    plt.tight_layout()
    plt.show()
    if data_args.do_save_visualize:
        xml = int(re.findall(r'\d+', sample['file_name'][0])[0])
        fig.savefig(os.path.join(output_dir, f'{index}_xml{xml}.png'))

def visualize_layout_task(sample, label_text, prediction_texts, input_text, data_args, output_dir, images, index):
    idx_ = int(re.findall(r'\d+', sample._file_name)[-1])
    image_path = os.path.join(data_args.data_dir , 'images', f'image_{idx_}.png')
    original_image = Image.open(image_path)
    images.insert(0, original_image)       # images = [original_image, recommeded_image1, 2 ...]
    assert len(prediction_texts) == len(images), "Prediction_texts and images do not match!"

    label_tokens = parse_token(label_text)
    input_tokens = parse_input(input_text)

    scale_factor = len(images)
    fig, axs = plt.subplots((scale_factor), 3, figsize=(15, 5*(scale_factor)))

    titles = ['Original']
    titles.extend([f'Top {i}' for i in range(1, (scale_factor-1))])
    titles.append('Blank')

    for i, (text, image) in enumerate(zip(prediction_texts, images)) :
        image = T.ToTensor()(image)
        prediction_tokens = parse_token(text)

        blank_canvas = torch.ones(image.shape, dtype=torch.float)
        masked_image = add_bbox_to_image(image, label_tokens, (0, 1, 0, 0.5))
        predicted_image = add_bbox_to_image(image, prediction_tokens, (0, 1, 0, 0.5))
        canvas_image = add_bbox_to_image(blank_canvas, prediction_tokens, (0, 1, 0, 0.5))

        # Plot the masked image using label layout
        axs[0].imshow(masked_image.permute(1, 2, 0))
        for idx, (input_token, label_token) in enumerate(zip(input_tokens, label_tokens)):
            x1, y1, x2, y2 = label_token['bbox']
            x1 = round(x1 * (masked_image.shape[2] - 1) / 500)
            x2 = round(x2 * (masked_image.shape[2] - 1) / 500)
            y1 = round(y1 * (masked_image.shape[1] - 1) / 500)
            y2 = round(y2 * (masked_image.shape[1] - 1) / 500)
            axs[0].text(x1, y1, idx, fontsize=8, bbox=dict(alpha=0.2))
        axs[0].set_title(f'{titles[i]} Masked (Label Layout)')

        # Plot the prediction masked image
        axs[1].imshow(predicted_image.permute(1, 2, 0))
        for idx, (input_token, prediction_token) in enumerate(zip(input_tokens, prediction_tokens)):
            x1, y1, x2, y2 = prediction_token['bbox']
            x1 = round(x1 * (predicted_image.shape[2] - 1) / 500)
            x2 = round(x2 * (predicted_image.shape[2] - 1) / 500)
            y1 = round(y1 * (predicted_image.shape[1] - 1) / 500)
            y2 = round(y2 * (predicted_image.shape[1] - 1) / 500)
            axs[1].text(x1, y1, idx, fontsize=8, bbox=dict(alpha=0.2))
        axs[1].set_title(f'{titles[i]} Prediction Layout')

        # Plot the prediction masked image
        axs[2].imshow(canvas_image.permute(1, 2, 0))
        for idx, (input_token, prediction_token) in enumerate(zip(input_tokens, prediction_tokens)):
            x1, y1, x2, y2 = prediction_token['bbox']
            x1 = round(x1 * (canvas_image.shape[2] - 1) / 500)
            x2 = round(x2 * (canvas_image.shape[2] - 1) / 500)
            y1 = round(y1 * (canvas_image.shape[1] - 1) / 500)
            y2 = round(y2 * (canvas_image.shape[1] - 1) / 500)
            axs[2].text(x1, y1, idx, fontsize=8, bbox=dict(alpha=0.2))
        axs[2].set_title(f'{titles[i]} Prediction Layout')

    label_text = '\n'.join([f"{token['id']}: {token['text']}" for token in input_tokens])

    plt.tight_layout()
    plt.show()
    if data_args.do_save_visualize:
        f = open(os.path.join(output_dir, f'{index}_xml{idx_}_text.txt'), 'w')
        f.write(label_text)
        f.close()
        fig.savefig(os.path.join(output_dir, f'{index}_xml{idx_}.png'))

def inference_layout_task(sample, prediction_texts, input_text, data_args, output_dir, images, index):
    original_image = images[0]
    assert len(prediction_texts) == len(images), "Prediction_texts and images do not match!"

    input_tokens = parse_input(input_text)

    scale_factor = len(images)
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))


    for i, (text, image) in enumerate(zip(prediction_texts, images)):
        image = T.ToTensor()(image)
        prediction_tokens = parse_token(text)

        blank_canvas = torch.ones(image.shape, dtype=torch.float)
        predicted_image = add_bbox_to_image(image, prediction_tokens, (0, 1, 0, 0.5))
        canvas_image = add_bbox_to_image(blank_canvas, prediction_tokens, (0, 1, 0, 0.5))

        # Plot the prediction masked image
        axs.imshow(canvas_image.permute(1, 2, 0))
        for idx, (input_token, prediction_token) in enumerate(zip(input_tokens, prediction_tokens)):
            x1, y1, x2, y2 = prediction_token['bbox']
            x1 = round(x1 * (canvas_image.shape[2] - 1) / 500)
            x2 = round(x2 * (canvas_image.shape[2] - 1) / 500)
            y1 = round(y1 * (canvas_image.shape[1] - 1) / 500)
            y2 = round(y2 * (canvas_image.shape[1] - 1) / 500)
            axs.text(x1, y1, idx, fontsize=8, bbox=dict(alpha=0.2))
        axs.set_title(f'Blank Prediction Layout')

    label_text = '\n'.join([f"{token['id']}: {token['text']}" for token in input_tokens])

    plt.tight_layout()
    plt.show()
    if data_args.do_save_visualize:
        f = open(os.path.join(output_dir, f'inference_{index}_text.txt'), 'w')
        f.write(label_text)
        f.close()
        fig.savefig(os.path.join(output_dir, f'inference_{index}.png'))
