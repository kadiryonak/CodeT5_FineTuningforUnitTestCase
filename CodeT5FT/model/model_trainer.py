from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from utils.metrics import Metrics
import torch

class CodeT5Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.training_args = TrainingArguments(
            output_dir='./results',
            dataloader_pin_memory=True,
            eval_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=60,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=10,
        )

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model)
        )

    def train(self):
        print("Model eğitilmeye başlandı...")
        self.trainer.train()
        print("Model eğitimi tamamlandı.")
        self.model.save_pretrained('./results')
        self.tokenizer.save_pretrained('./results')

    def evaluate(self):
        print("Model değerlendiriliyor...")
        metrics = self.trainer.evaluate()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.cpu().numpy()
        print("Değerlendirme sonuçları:", metrics)
        return metrics

    def evaluate_test_metrics(self, test_dataset):
        print("Test verisi üzerinde BLEU ve CodeBLEU hesaplanıyor...")
        metrics = Metrics.evaluate_code_metrics(test_dataset, self.model, self.tokenizer)
        print("Test Metrikleri:", metrics)
        return metrics
