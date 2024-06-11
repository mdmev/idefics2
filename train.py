from src.config import Config
from src.dataset import CustomDataset
from src.collator import MyDataCollator
from src.model import ModelLoader
from src.trainer import ModelTrainer

def main():
    processor = ModelLoader.load_processor("HuggingFaceM4/idefics2-8b-chatty")
    model = ModelLoader.load_model(
        "HuggingFaceM4/idefics2-8b-chatty",
        Config.USE_QLORA,
        Config.USE_LORA,
        Config.DTYPE,
    )

    train_dataset = CustomDataset(Config.JSON_PATH, split="train")
    eval_dataset = CustomDataset(Config.JSON_PATH, split="test")
    data_collator = MyDataCollator(processor)

    training_args = ModelTrainer.create_training_args(Config.OUTPUT_DIR)
    ModelTrainer.train_model(model, training_args, data_collator, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
