"""
Simple test script to check if CUDA is available for generating text and training models
"""

import torch

def test_cuda() -> None:
    """
    Check to see if CUDA available on system

    Args:
        None
    
    Returns:
        None
    """
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    test_cuda()