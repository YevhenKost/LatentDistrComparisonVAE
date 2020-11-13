import random
import numpy as np
import torch


def _fix_seeds():
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.backends.cudnn.deterministic = True
    random.seed(2)