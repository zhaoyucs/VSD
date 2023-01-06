# Feature extraction

We use [Hao Tan's Detectron2 implementation of 'Bottom-up feature extractor'](https://github.com/airsplay/py-bottom-up-attention), which is compatible with [the original Caffe implementation](https://github.com/peteanderson80/bottom-up-attention).

Following LXMERT, we use the feature extractor which outputs 36 boxes per image.
We store features in hdf5 format.


## Install feature extractor (optional)

Please follow [the original installation guide](https://github.com/airsplay/py-bottom-up-attention#installation).

## Manually extract & convert features (optional)

* `_prpoposal.py`: extract features from 36 detected boxes
* `_gt.py`: extract features from ground truth boxes
