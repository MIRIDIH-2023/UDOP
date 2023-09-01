#!/usr/bin/env python
# coding=utf-8

import json
import os
import shutil
import tempfile
import unittest
from typing import List

import numpy as np

from core.transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast, UdopForConditionalGeneration, UdopConfig
from core.transformers.models.udop import UdopTokenizer, UdopTokenizerFast
from core.transformers.testing_utils import (
    require_pytesseract,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)
from core.transformers.utils import FEATURE_EXTRACTOR_NAME, cached_property, is_pytesseract_available, is_torch_available


if is_torch_available():
    import torch


if is_pytesseract_available():
    from PIL import Image

    from core.transformers import UdopImageProcessor, UdopProcessor


image_processor = UdopImageProcessor()

model = UdopForConditionalGeneration.from_pretrained("nielsr/udop-large")
tokenizer = UdopTokenizer.from_pretrained("ArthurZ/udop", model_max_length=512)
processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)


images = Image.open('data/images/image_0.png').convert("RGB")


task = "Layout Modeling."
encoding = processor(images=images, text=task, return_tensors="pt")

input_ids = encoding.input_ids
bbox = encoding.bbox.float()
pixel_values = encoding.pixel_values


# single forward pass
print("Testing single forward pass..")
with torch.no_grad():
    decoder_input_ids = torch.tensor([[101]])
    outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
    print("Shape of logits:", outputs.logits.shape)
    print("First values of logits:", outputs.logits[0, :3, :3])

# autoregressive decoding
print("Testing generation...")
model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
predicted_ids = model.generate(**encoding)
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])