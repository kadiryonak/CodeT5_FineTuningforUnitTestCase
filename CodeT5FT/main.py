import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

# Senin yazdığın modüller
from model.model_trainer import CodeT5Trainer
from data.dataset import JsonDataReader, TestCase, DataPreparer, CustomDataset
from model import CodeT5TokenizerLoader, CodeT5ModelLoader, CodeT5Manager
from utils.metrics import Metrics

if __name__ == '__main__':

    # Cuda'yı kullanma
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer ve Model Yükleme
    model_name = "Salesforce/codet5-base"
    tokenizer_loader = CodeT5TokenizerLoader(model_name)
    model_loader = CodeT5ModelLoader(model_name)

    tokenizer, model = CodeT5Manager(tokenizer_loader, model_loader).load_tokenizer_and_model()

    # LoRA konfigürasyonu
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )

    # LoRA modelini oluşturma
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Modeli GPU'ya taşıma
    model = model.to(device)

    # Veri Yolu
    data_path = r"C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld\TestCaseCodeT5\methods2test_small"

    # Veri Okuma
    json_reader = JsonDataReader(data_path)
    train_data, eval_data, test_data = json_reader.read_data()

    # Analiz
    test_case = TestCase(json_reader)
    features = test_case.analyze()

    # Veri Hazırlama
    data_preparer = DataPreparer(features, tokenizer)
    train_data, eval_data, test_data = data_preparer.prepare_data()

    # Tensor'leri GPU'ya taşıma
    for split in [train_data, eval_data, test_data]:
        split['input_ids'] = split['input_ids'].to(device)
        split['attention_masks'] = split['attention_masks'].to(device)
        split['labels'] = split['labels'].to(device)

    # Datasetleri oluşturma
    train_dataset = CustomDataset(train_data['input_ids'], train_data['attention_masks'], train_data['labels'])
    eval_dataset = CustomDataset(eval_data['input_ids'], eval_data['attention_masks'], eval_data['labels'])
    test_dataset = CustomDataset(test_data['input_ids'], test_data['attention_masks'], test_data['labels'])

    # Eğitici sınıfını başlat
    trainer = CodeT5Trainer(model, tokenizer, train_dataset, eval_dataset)

    # Modeli eğit ve değerlendir
    trainer.train()
    trainer.evaluate()

    # Test verisi için BLEU ve CodeBLEU hesapla
    code_metrics = trainer.evaluate_test_metrics(test_dataset)

    print("Kod Değerlendirme Metrikleri:", code_metrics)
