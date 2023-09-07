from dataclasses import dataclass, field
from typing import Optional

import torch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image
from transformers import HfArgumentParser

from core.common.utils import (get_visual_bbox, img_trans_torchvision,
                               str_to_img)
from core.models import (UdopConfig, UdopTokenizer,
                         UdopUnimodelForConditionalGeneration)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_name: Optional[str] = field(
        default="Layout Modeling", 
        metadata={"help": "The name of the task (Layout Modeling, Visual Text Recognition, Joint Text-Layout Reconstruction, All)."}
    )
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
    
parser = HfArgumentParser(Arguments)
args = parser.parse_yaml_file(yaml_file="config/api.yaml")[0]

config = UdopConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=True if args.use_auth_token else None,
    attention_type=args.attention_type if args.attention_type else None,
)

tokenizer = UdopTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=True,
    revision=args.model_revision,
    use_auth_token=True if args.use_auth_token else None,
)

model = UdopUnimodelForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=True if args.use_auth_token else None,
)


# Test route
@app.route('/')
def index():
    return 'UDOP API'

# TODO: Add padding to input_ids, token_boxes
@app.route('/main', methods=['POST'])
def main():
    response = Response()
    if request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
        data = request.get_json()

        # text: List[str]
        # Assumes text is given as a list of sentences
        # ex) text = ['Sentence 1', 'Sentence 2']
        text = data['text']

        sentinel_idx = 0
        masked_text = []
        word_tokens = []
        task_boxes = []
        token_boxes = []

        for sentence in text:
            masked_text.append(f'<extra_l_id_{sentinel_idx}>')
            masked_text.append(sentence)
            masked_text.append(f'</extra_l_id_{sentinel_idx}>')

            if sentinel_idx > 100:
                break

            sentinel_idx += 1

        task_tokens = tokenizer.tokenize(args.task_name + ".")
        task_boxes = [[0,0,0,0] for _ in range(len(task_tokens))]
        for text in masked_text:
            word_tokens.extend(token for token in (tokenizer.tokenize(text)))
        token_boxes = [[0,0,0,0] for _ in range(len(word_tokens))]

        task_ids = tokenizer.convert_tokens_to_ids(task_tokens)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        # Prepare model inputs
        input_ids = task_ids + word_ids
        seg_data = task_boxes + token_boxes
        attention_mask = [1] * len(input_ids)
        image = str_to_img(data['image']) if 'image' in data else Image.new('RGB', (224, 224), 'rgb(0, 0, 0)')
        image = img_trans_torchvision(image, 224)
        visual_seg_data = get_visual_bbox() 

        # Process and batch inputs
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        seg_data = torch.tensor(seg_data, dtype=torch.float).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)
        visual_seg_data = visual_seg_data.unsqueeze(0).to(device)

        # Generate output
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

        prediction_text = tokenizer.decode(output_ids[0][1:-1])

        data = {}
        data['prediction'] = prediction_text
        response = jsonify(data)
        return response


if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=5000)