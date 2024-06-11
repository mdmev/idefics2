import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

# Add the 'src' directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Import classes and functions from your modules
from config import Config
from dataset import CustomDataset
from collator import MyDataCollator
from model import ModelLoader

def eval(model, example, processor, num_samples=None, device=Config.DEVICE):
    model.eval()
    images = []

    image = example["image"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            ]
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images.append(image)

    inputs = processor(
        text=[text], 
        images=images,
        return_tensors="pt",
        padding=True,
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # print(f"Generated text: {text}")
    # for i in inputs.keys():
    #     print(f"\n\n\nProcessed inputs: {inputs[i].shape if isinstance(inputs[i], torch.Tensor) else None}\n\n")
    # print(inputs)

    return generated_texts


# Main function to load the model and evaluate
def main():
    processor = ModelLoader.load_processor("HuggingFaceM4/idefics2-8b-chatty")
    model = ModelLoader.load_model(
        Config.CHECKPOINT,
        Config.USE_QLORA,
        Config.USE_LORA,
        Config.DTYPE,
        Config.CUSTOM_TEMP_DIR
    )
    

    eval_dataset = CustomDataset(Config.JSON_PATH, split="test")
    example = eval_dataset[1]
    result = eval(model, example, processor, num_samples=1, device=Config.DEVICE)
    print(f"Result for single image: {result}")
    print("\n")
    print(example["image"])
    print(example['text'])

if __name__ == "__main__":
    main()
