#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import torch
import transformers
from PIL import Image
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.models.udop import (
    UdopConfig,
    UdopForConditionalGeneration,
    UdopImageProcessor,
    UdopProcessor,
    UdopTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from core.common.utils import inference_layout_task, random_split, visualize_layout_task
from core.datasets import MIRIDIH_Dataset
from core.trainers import CurriculumTrainer, DataCollator, elevateMRCallback

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    unit: Optional[str] = field(
        default="word", metadata={"help": "The unit of tokenize (word, token)."}
    )
    curriculum: Optional[str] = field(
        default=False,
        metadata={"help": "The choice of curriculum learning (True or False)."},
    )
    curri_patience: Optional[int] = field(
        default=None, metadata={"help": "Number of times it was not been updated"}
    )
    curri_threshold: Optional[int] = field(
        default=None,
        metadata={"help": "Criteria for determining that an update has been made"},
    )
    curri_start_MR: Optional[int] = field(
        default=None,
        metadata={
            "help": "The starting point of masking ratio from curri_start_MR to 100%"
        },
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "local dataset stored location"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
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
        metadata={"help": "image size" "value if set."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length_decoder: int = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_save_visualize: bool = field(
        default=False,
        metadata={"help": "Whether to save visualizations in predict"},
    )
    do_inference: bool = field(
        default=False,
        metadata={"help": "Whether to inference model"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
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
        metadata={
            "help": "Attention type: BigBird configuruation only. Choices: block_sparse (default) or original_full"
        },
    )
    loss_fct: str = field(
        default="CE",
        metadata={"help": "Loss function for location tokens. Default: None"},
    )


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # If error, modify training_args.py "_frozen"
    training_args.logging_dir = os.path.join(training_args.output_dir, "runs")
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, "cache")
    if training_args.do_train:
        os.makedirs(model_args.cache_dir, exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

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

    # Load processo,r pretrained model and tokenizer

    image_processor = UdopImageProcessor(
        apply_ocr=False,
        size={"height": data_args.image_size, "width": data_args.image_size},
    )
    tokenizer = UdopTokenizer.from_pretrained("ArthurZ/udop")
    config = UdopConfig.from_pretrained("nielsr/udop-large")
    model = UdopForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config
    ).to(device)
    processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Load datasets
    total_dataset = MIRIDIH_Dataset(
        processor=processor, tokenizer=tokenizer, data_args=data_args
    )

    train_dataset, eval_dataset, test_dataset = random_split(
        total_dataset, [0.8, 0.1, 0.1]
    )

    # Data collator
    padding = "max_length" if data_args.pad_to_max_length else False
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        max_length_decoder=data_args.max_seq_length_decoder,
    )

    def preprocess_logits_for_metrics(logits, labels):
        return logits[0].argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)

    # Used to adjust masking ratio when using curriculum learning
    elevateMRcallback = elevateMRCallback(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        early_stopping_patience=data_args.curri_patience,
        early_stopping_threshold=data_args.curri_threshold,
    )

    # Initialize our Trainer
    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[elevateMRcallback] if data_args.curriculum else None,
        data_collator=data_collator,
        loss_fct=model_args.loss_fct,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(test_dataset)
        )
        metrics["test_samples"] = min(max_val_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Predict on test dataset
    if training_args.do_predict:
        logger.info("*** Predict ***")
        os.makedirs(training_args.output_dir, exist_ok=True)

        test_dataset.dataset.set_layout_modeling_masking_ratio(1.0)

        while True:
            idx = input(f"Enter idx (or 'quit') in range 0 ~ {len(test_dataset)-1}: ")
            if idx.lower() == "quit":
                break

            encoding = test_dataset.__getitem__(int(idx))

            # Process and batch inputs
            encoding["input_ids"] = (
                torch.tensor(encoding["input_ids"], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            encoding["bbox"] = (
                torch.tensor(encoding["bbox"], dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            encoding["attention_mask"] = (
                torch.tensor(encoding["attention_mask"], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            encoding["pixel_values"] = encoding["pixel_values"].unsqueeze(0).to(device)
            encoding["labels"] = (
                torch.tensor(encoding["labels"], dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )

            predicted_ids = model.generate(**encoding, num_beams=1, max_length=512)

            input_text = tokenizer.decode(encoding["input_ids"][0])
            prediction_text = processor.decode(predicted_ids[0][1:-1])
            label_text = processor.batch_decode(encoding["labels"])[0]
            images = []

            visualize_layout_task(
                encoding,
                label_text,
                [prediction_text],
                input_text,
                data_args,
                training_args.output_dir,
                images,
                idx,
            )

            print("Input: ", input_text)
            print("\nLabel: ", label_text)
            print("\nPrediction: ", prediction_text)
            print()

    # Inference on raw text and image
    if data_args.do_inference:
        logger.info("*** Inference ***")

        os.makedirs(training_args.output_dir, exist_ok=True)

        text = [
            "2023 DesignCat Annual Report",
            "Sales department report",
            "Sangyun",
            "sangyun0914@gmail.com",
        ]
        image = Image.new("RGB", (224, 224), "rgb(0, 0, 0)")

        # Prepare model inputs
        sentinel_idx = 0
        masked_text = []
        token_boxes = []

        # Mask sentences
        for sentence in text:
            masked_text.append(f"<extra_l_id_{sentinel_idx}>")
            masked_text.append(sentence)
            masked_text.append(f"</extra_l_id_{sentinel_idx}>")

            if sentinel_idx > 100:
                break
            sentinel_idx += 1

        token_boxes = [[0, 0, 0, 0] for _ in range(len(masked_text))]

        # Encode inputs
        encoding = processor(
            images=image,
            text=["Layout Modeling."],
            text_pair=[masked_text],
            boxes=[token_boxes],
            return_tensors="pt",
        )
        encoding = {key: value.to(device) for key, value in encoding.items()}

        # Generate layout predictions
        predicted_ids = model.generate(**encoding, num_beams=1, max_length=512)

        input_text = tokenizer.decode(encoding["input_ids"][0])
        prediction_text = processor.decode(predicted_ids[0][1:-1])
        images = [image]

        inference_layout_task(
            encoding,
            [prediction_text],
            input_text,
            data_args,
            training_args.output_dir,
            images,
            "0",
        )

        print("Input: ", input_text)
        print("\nPrediction: ", prediction_text)
        print()


if __name__ == "__main__":
    main()
