import torch

class DataPreparer:
    def __init__(self, features, tokenizer, max_length=512, device="cpu"):
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def prepare_data(self):
        datasets = {
            'train': {'input_ids': [], 'attention_masks': [], 'labels': []},
            'eval': {'input_ids': [], 'attention_masks': [], 'labels': []},
            'test': {'input_ids': [], 'attention_masks': [], 'labels': []}
        }

        if self.features is None:
            raise ValueError("Hata: features None. Özellikler doğru yüklenmedi.")

        for dataset_name, features in self.features.items():
            for feature in features:
                input_text = f"{feature['focal_context_4']}"
                target_text = feature['target']

                input_encodings = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                target_encodings = self.tokenizer(
                    target_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                datasets[dataset_name]['input_ids'].append(input_encodings['input_ids'])
                datasets[dataset_name]['attention_masks'].append(input_encodings['attention_mask'])
                datasets[dataset_name]['labels'].append(target_encodings['input_ids'])

        # Tensorları birleştir
        for dataset_name in datasets:
            datasets[dataset_name]['input_ids'] = torch.cat(datasets[dataset_name]['input_ids'], dim=0).to(self.device)
            datasets[dataset_name]['attention_masks'] = torch.cat(datasets[dataset_name]['attention_masks'], dim=0).to(self.device)
            datasets[dataset_name]['labels'] = torch.cat(datasets[dataset_name]['labels'], dim=0).to(self.device)

        return datasets['train'], datasets['eval'], datasets['test']
