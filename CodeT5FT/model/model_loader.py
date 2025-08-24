from transformers import T5ForConditionalGeneration

class CodeT5ModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        print(f"Model yükleniyor: {self.model_name}")
        return T5ForConditionalGeneration.from_pretrained(self.model_name)
