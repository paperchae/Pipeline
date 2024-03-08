import os
import torch

from utils.config import get_config

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True


def main():
    config = get_config('../config.yaml')

    # Set Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device count: {torch.cuda.device_count()}")

    # Load Datasets

