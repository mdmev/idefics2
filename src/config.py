import os
import torch

class Config:
    USE_QLORA = False
    USE_LORA = False
    DTYPE = torch.bfloat16
    CHECKPOINT= ""
    DEVICE_MAP = "FSDP"
    JSON_PATH = "../test_dataset.json"
    OUTPUT_DIR = ""
    DEVICE="cuda"
