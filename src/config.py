#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import src

REPO_PATH: Path = Path(src.__path__[0]).parent
DATA_PATH: Path = REPO_PATH / 'c-FOS'
MODEL_PATH: Path = REPO_PATH / 'src' / 'model' / 'pretrained_model' / 'c-resunet_y_11_dec_bottl.h5'