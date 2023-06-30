#!/usr/bin/env python
# coding=utf-8
from PIL import Image
from core.common.utils import img_trans_torchvision, get_visual_bbox

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer, 
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from core.datasets import MIRIDIH_Dataset
from core.trainers import DataCollator
from core.models import UdopUnimodelForConditionalGeneration, UdopConfig, UdopTokenizer


MODEL_CLASSES = {
    'UdopUnimodel': (UdopConfig, UdopUnimodelForConditionalGeneration, UdopTokenizer),
}

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "local dataset stored location"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    image_size: Optional[int] = field(
    default=512,
    metadata={
        "help": "image size"
        "value if set."
    },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )   
    max_seq_length_decoder: int = field(
        default=16,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )    


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        default=None, metadata={'help': 'Model type selected in the list.'})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )    
    attention_type: str = field(
        default="original_full",
        metadata={"help": "Attention type: BigBird configuruation only. Choices: block_sparse (default) or original_full"},
    )


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    config = UdopConfig.from_pretrained("finetune_miridih")
    tokenizer = UdopTokenizer.from_pretrained("finetune_miridih")
    model = UdopUnimodelForConditionalGeneration.from_pretrained("finetune_miridih")


   # Get datasets
    total_dataset = (MIRIDIH_Dataset(data_args=data_args, tokenizer=tokenizer)
                     if training_args.do_train else None)

    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.8, 0.1, 0.1])

    x = test_dataset.__getitem__(0)

    input_ids = torch.unsqueeze(x['input_ids'], dim=0) 
    labels = x['labels']
    seg_data = torch.unsqueeze(x['seg_data'], dim=0) 
    im = torch.unsqueeze(x['image'], dim=0)  
    visual_seg_data = torch.unsqueeze(x['visual_seg_data'], dim=0)

    label_text = tokenizer.decode(labels)

    output_ids = model.generate(
            input_ids,
            seg_data=seg_data,
            image=im,
            visual_seg_data=visual_seg_data,
            use_cache=True,
            decoder_start_token_id=None,
            num_beams=1,
            max_length=10,
        )
    output_text = tokenizer.decode(output_ids[0][1:-1])
    print("Prediction: " + output_text)
    print("label: " + label_text)



if __name__ == "__main__":
    main()

