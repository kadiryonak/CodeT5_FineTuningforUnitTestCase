import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels, device="cpu"):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx].clone().detach().to(self.device)
        attention_mask = self.attention_masks[idx].clone().detach().to(self.device)
        label = self.labels[idx].clone().detach().to(self.device)

        return {
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'labels': label
        }
