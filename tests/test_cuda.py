"""
Simple test script to check if CUDA is available for generating text and training models
"""

import torch

def test_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    test_cuda()