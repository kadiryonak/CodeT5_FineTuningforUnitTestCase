from transformers import RobertaTokenizer

class CodeT5TokenizerLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_tokenizer(self):
        print(f"Tokenizer yükleniyor: {self.model_name}")
        return RobertaTokenizer.from_pretrained(self.model_name)
