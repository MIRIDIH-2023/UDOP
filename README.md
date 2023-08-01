# Unifying Vision, Text, and Layout for Universal Document Processing
## Finetuning UDOP model for recommendation and modification of PowerPoint template

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)


## Project Overview![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)
---

This repository contains the source code for 2023 MIRIDIH Corporate Collaboration Project. This project utilises UDOP as baseline model to recommend and further modify a PPT template based on user's query.

## Install
---
### Setup `python` environment
```
conda create -n UDOP python=3.8   # You can also use other environment.
```
### Install other dependencies
```
pip install -r requirements.txt
```

## Repository Structure
---
``` bash
.
├── LICENSE
├── README.md
├── SBERT/                           # Integrate SBERT for template recommendation
│   ├── embedded/
│   └── sbert_keyword_extractor_2023_07_18/
├── config/                          # Train/Inference configuration files
│   ├── inference.yaml
│   └── train.yaml
├── core/                            # Main source code
│   ├── common/
│   ├── datasets/
│   ├── models/
│   └── trainers/
├── data/                            # Custom dataset folder
│   ├── images/
│   │   └── image_{idx}.png
│   ├── json_data/
│   │   └── processed_{idx}.pickle
│   └── SBERT/
│       └── json_data2/
├── main.py                     
├── models                          # Trained models saved to this folder
├── test                            # Save visualizations during inference
├── requirements.txt
├── udop-unimodel-large-224         # Pretrained UDOP model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
└── utils                           # Utilities
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