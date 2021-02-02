import random
import numpy as np
import torch

def encoding(tokens, encoding = 'utf-8'):
    
    tokens = [token.encode(encoding) for token in tokens]
    return tokens

def _fix_seeds():
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.backends.cudnn.deterministic = True
    random.seed(2)
