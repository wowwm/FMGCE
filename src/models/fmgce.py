# For academic security reasons, FMGCE's code has not been officially made public yet.
# To foster reproducible research, our code will be made publicly available at https://github.com/wowwm/FMGCE

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class FMGCE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FMGCE, self).__init__(config, dataset)
