<div align="center">

# Design Layout Generation through <br> Vision-Text-Layout Transformer

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>

Baseline - [Unifying Vision, Text, and Layout for Universal Document Processing <br> (CVPR 2023 Highlight)](https://arxiv.org/abs/2212.02623)

## Project Overview![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)

<img src="https://github.com/miridi-sanhak/UDOP/assets/96368116/c7c2a63f-ba1b-43e3-ba4d-9d20cd91147b">

The goal of the 2023 MIRIDIH Corporate Collaboration Project is to utilise MIRIDIH's dataset to recommend and generate an optimal design layout for a given user's query. This repository contains the code responsible for generating the design layout, specifically to generate bounding box tokens for each sentence in the query. This repository contains the code responsible for following tasks:
- Preprocess raw XML data to extract text, image and layout data
- Finetune Vision-Text-Layout(VTL) Transformer, based on Universal Document Processing (UDOP) model for design layout generation.
- Inference VTL model to generate bounding box tokens for each sentence in the query


## Model Architecture
The Vision-Text-Layout Transformer is based on UDOP, a model proposed in [Unifying Vision, Text, and Layout for Universal Document Processing (CVPR 2023 Highlight)](https://arxiv.org/abs/2212.02623). UDOP is a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation.
We have modified and fine-tuned the UDOP model for design layout generation. The model architecture is as follows:





## Repository Structure
The repository has two branches:
- `main` branch contains the code for customizing HuggingFace's implmentation of the UDOP model
- `custom` branch contains the code for customizing Microsoft's implmentation of the UDOP model


``` bash
.
├── LICENSE
├── README.md
├── config/                          # Train/Inference configuration files
│   ├── inference.yaml
│   ├── predict.yaml
│   └── train.yaml
├── core/                            # Main UDOP/DataClass source code
│   ├── common/
│   ├── datasets/
│   ├── models/
│   └── trainers/
├── data/                            # Custom dataset folder
│   ├── images/
│   │   └── image_{idx}.png
│   └── json_data/
│       └── processed_{idx}.pickle
├── main.py                         # Main script to run training/inference
├── models                          # Trained models saved to this folder
├── sweep.py                        # Script to run hyperparameter sweep
├── test                            # Save visualizations during inference
├── requirements.txt
├── udop-unimodel-large-224         # Pretrained UDOP 224 model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
├── udop-unimodel-large-512         # Pretrained UDOP 512 model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
└── utils                           # Utilities
```

## Install
### Setup `python` environment
```
conda create -n UDOP python=3.8   # You can also use other environment.
```
### Install other dependencies
```
pip install -r requirements.txt
```



## Scripts
Setup folder structures as above and modify config/ yaml files for customization

### Finetune UDOP model
```
python main.py config/train.yaml
```

### Inference UDOP model
```
python main.py config/inference.yaml
```