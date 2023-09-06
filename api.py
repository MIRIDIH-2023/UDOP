from dataclasses import dataclass, field
from typing import Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from transformers import HfArgumentParser

from core.models import (UdopConfig, UdopTokenizer,
                         UdopUnimodelForConditionalGeneration)

app = Flask(__name__)
CORS(app)

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


@app.route('/main', methods=['POST'])
def main():
    response = Response()
    if request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
        data = request.get_json()
        print(data)







if __name__ == '__main__':    
    app.run(host="0.0.0.0", port=8000)