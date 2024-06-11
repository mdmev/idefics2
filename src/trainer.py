from transformers import TrainingArguments, Trainer
import mlflow
import mlflow.pytorch
import os

class ModelTrainer:
    @staticmethod
    def create_training_args(output_dir):
        return TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-7,
            weight_decay=0.01,
            logging_steps=1,
            output_dir=output_dir,
            eval_steps=100,        #idk
            save_strategy="steps", #steps
            save_steps=200,        #100 mayb
            save_total_limit=5,    #3?
            eval_strategy="steps", #steps
            bf16=True,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': True},
            # report_to=["wandb","mlflow"],
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    @staticmethod
    def train_model(model, training_args, data_collator, train_dataset, eval_dataset):        
        with mlflow.start_run() as run:
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            trainer.train()


            mlflow.log_params(training_args.to_sanitized_dict())
