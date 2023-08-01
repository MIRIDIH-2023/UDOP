#!/usr/bin/env python
# coding=utf-8

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import evaluate
import numpy as np
import requests
import torch
import transformers
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from core.common.utils import (img_trans_torchvision, random_split,
                               visualize_layout_task,
                               visualize_text_layout_task, visualize_text_task)
from core.datasets import MIRIDIH_Dataset
from core.models import (UdopConfig, UdopTokenizer,
                         UdopUnimodelForConditionalGeneration)
from core.trainers import DataCollator
from SBERT.model import SBERT
from SBERT.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    unit: Optional[str] = field(default="word", metadata={"help": "The unit of tokenize (word, token)."})
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
    do_save_visualize: bool = field(
        default=False,
        metadata={
            'help':'Whether to save visualizations in predict'
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
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs')
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, 'cache')
    if training_args.do_train:
        os.makedirs(model_args.cache_dir, exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

 
    #if 'local' in model_args.model_name_or_path:
    if model_args.model_type in MODEL_CLASSES:
        config_type, model_type, tokenizer_type = MODEL_CLASSES[model_args.model_type]
    else:
        config_type, model_type, tokenizer_type = AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = config_type.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        attention_type=model_args.attention_type if model_args.attention_type else None,
    )

    tokenizer = tokenizer_type.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

   # Get datasets
    total_dataset = MIRIDIH_Dataset(data_args=data_args, tokenizer=tokenizer)

    train_dataset, eval_dataset, test_dataset = random_split(total_dataset, [0.8, 0.1, 0.1])

    # Data collator
    padding = "max_length" if data_args.pad_to_max_length else False
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        max_length_decoder=data_args.max_seq_length_decoder,
    )
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        os.makedirs(training_args.output_dir, exist_ok=True)

        model_path='./SBERT/sbert_keyword_extractor_2023_07_18' #모델 저장 경로
        sbert_model = SentenceTransformer(model_path)
        sbert_model = SBERT(sbert_model)
        sbert_model.load_emedding_vector(save_path="./SBERT/embedded",file_name="keyword_embedding_list.pickle") #load vector for get_keyword

        while True:
            idx = input(f"Enter idx (or 'quit') in range 0 ~ {len(test_dataset)-1}: ")
            if idx.lower() == "quit":
                break

            sample = data_collator([test_dataset.__getitem__(int(idx))])

            input_ids = sample['input_ids'].to(device)
            labels = sample['labels'].to(device)
            seg_data = sample['seg_data'].to(device)
            im = sample['image'].to(device)
            visual_seg_data = sample['visual_seg_data'].to(device)

            save_im_to_generate = [im]
            images = []                     # used for saving recommended, and blank image

            if use_text_image_only:=True:   # Text only, image added
                K = 2
                K = int(input("Enter the number of images to be recommended in range 1 ~ 5: "))
                while (K < 1 or K > 5) :
                    K = int(input("Wrong number! \nPlease enter the number in range 1 ~ 5: "))

                # set seg_data (bounding box) 0
                seg_data = torch.zeros((input_ids.shape[0],input_ids.shape[1], 4), device=input_ids.device, dtype=torch.float)

                # SBERT get image
                input_text_raw = tokenizer.decode(input_ids[0])
                processed_text = convert_to_sbert_input(input_text_raw)
                recommended = sbert_model.get_top_keyword(processed_text,top_k=K) #--> [ {'json','keyword','thumbnail_url','sheet_url'}, {}... ]
                
                # save recommended image and image used to predict label
                for rc in recommended :
                    try:
                        im = Image.open(BytesIO(requests.get(rc['thumbnail_url']).content)) # Need update for top k
                    except:
                        print(f"Error at loading image, {rc['thumbnail_url']}")
                        im = torch.zeros(im.shape, device=input_ids.device, dtype=torch.float)

                    images.append(im)
                    im = img_trans_torchvision(im, 224)
                    im = torch.unsqueeze(im, dim=0).to(device)
                    save_im_to_generate.append(im)

                # save blank image 
                im = Image.new('RGB', (500, 500), 'rgb(0, 0, 0)') 
                images.append(im)
                
                # save blank image to predict label
                im = img_trans_torchvision(im, 224)
                im = torch.unsqueeze(im, dim=0).to(device)
                save_im_to_generate.append(im)

            save_predicted_text = []
            for image in save_im_to_generate:
                output_ids = model.generate(
                        input_ids,
                        seg_data=seg_data,
                        image=image,
                        visual_seg_data=visual_seg_data,
                        use_cache=True,
                        decoder_start_token_id=tokenizer.pad_token_id,
                        num_beams=1,
                        max_length=512,
                    )
                
                print('generate complete!')
                prediction_text = tokenizer.decode(output_ids[0][1:-1])
                save_predicted_text.append(prediction_text)
                
            input_text = tokenizer.decode(input_ids[0])                # text in layout 
            
            label_list = labels[0].tolist()
            label_list = label_list[:label_list.index(1)+1] if 1 in label_list else label_list
            label_text = tokenizer.decode(label_list)                  # correct layout 

            if input_text.startswith("Layout Modeling"):
                visualize_layout_task(sample, label_text, save_predicted_text, input_text, data_args, training_args.output_dir, images, idx)
            elif input_text.startswith("Visual Text Recognition"):
                visualize_text_task(sample, label_text, prediction_text, input_text, data_args, training_args.output_dir, idx)
            elif input_text.startswith("Joint Text-Layout Reconstruction"):
                visualize_text_layout_task(sample, label_text, prediction_text, data_args, training_args.output_dir, idx)
            
            print("Input: ", input_text)
            print("\nLabel: ", label_text)
            print("\nPrediction: ", save_predicted_text)
            print()

def convert_to_sbert_input(s):
    matches = re.findall(r'<extra_l_id_\d+>(.*?)</extra_l_id_\d+>', s)
    cleaned_matches = [match.strip() for match in matches]
    return '\n'.join(cleaned_matches) + '\n'

if __name__ == "__main__":
    main()