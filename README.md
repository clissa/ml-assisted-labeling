# ML-Assisted pre-labeling for VGG Visual Image Annotator


[![GitHub license](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/clissa/ml-assisted-labeling/blob/main/LICENSE)
[![GitHub commit](https://img.shields.io/github/last-commit/clissa/ml-assisted-labeling)](https://github.com/clissa/ml-assisted-labeling)

This repository contains some utils to perform ML-assisted labeling compatible with 
[VGG Visual Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) (VIA).

To do this, we need two components:
 - pre-labels: the annotations we want to import into VIA 
 - converter: converts from binary masks to _**polygon annotations**_ compatible with VIA **_csv format_**

### Pre-labels

For the first point we can exploit a pre-trained model that performs binary segmentation and produces binary masks 
(see `ml-assisted-labeling.py`). 
This is application-specific, so you will need to either adapt the inference mode of your model or start directly
from binary masks and proceed with the next step.

### Converter
The binary masks are then converted into csv format compatible with VGG VIA (see `VGG_VIA_annotations.py`).
In particular, we target segmentation applications using **polygon masks**.
