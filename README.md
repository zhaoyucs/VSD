## Visual Description Description
* The datasets VSDv2 are available now.
<!-- <a href="https://github.com/unikcc/DiaASQ">
  <img src="https://img.shields.io/badge/DiaASQ-0.1-blue" alt="DiaASQ">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.8.1-green" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-orange" alt="Transformers">
</a>
<a href="https://github.com/unikcc/DiaASQ/blob/master/LICENSE" rel="nofollow">
  <img src="https://img.shields.io/badge/LICENSE-MIT-cyan" alt="LICENSE">
</a> -->

This repository cotains code and data for our paper [Visual Spatial Description: Controlled Spatial-Oriented Image-to-Text Generation](https://arxiv.org/abs/2210.11109)

** Note **
Please go into [VLT5](https://github.com/j-min/VL-T5) and follow the README there for Pretrained Models and Feature Extraction.


## Setup
```bash
# Create python environment (optional)
conda create -n vsd python=3.7
source activate vsd

# Install python dependencies
pip install -r requirements.txt

# For captioning evaluation
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Code structure
```bash
# Store images, features, and annotations
./datasets

# Image feature extraction
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        caption_sp.py, vrd_caption.py                         <= fine-tuning
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
```


## Pretrained Models
- pretrained VL-BART and VL-T5 are provided by [1]
- Download `snap/` from [Google Drive](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing)
```bash
gdrive download 1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph --recursive
```

## Run
```bash
bash ./baseline.sh gpu_num
bash ./end2end.sh gpu_num
```

## Acknowledgement

This repo is adapted from [VLT5](https://github.com/j-min/VL-T5).


## Reference
Please cite our paper if you use our models or data in your project.

```bibtex
@inproceedings{zhao2022vsd,
  title     = {Visual Spatial Description: Controlled Spatial-Oriented Image-to-Text
               Generation},
  author    = {Yu Zhao and
               Jianguo Wei and
               Zhichao Lin and
               Yueheng Sun and
               Meishan Zhang and
               Min Zhang},
  booktitle = {EMNLP},
  year      = {2022}
}
```