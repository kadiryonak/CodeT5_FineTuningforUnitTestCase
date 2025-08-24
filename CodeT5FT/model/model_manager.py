from .tokenizer_loader import CodeT5TokenizerLoader
from .model_loader import CodeT5ModelLoader
from peft import get_peft_model

class CodeT5Manager:
    def __init__(self, model_name: str, lora_config=None):
        self.model_name = model_name
        self.lora_config = lora_config

    def load_tokenizer_and_model(self):
        tokenizer = CodeT5TokenizerLoader(self.model_name).load_tokenizer()
        model = CodeT5ModelLoader(self.model_name).load_model()

        if self.lora_config:
            print("LoRA modeli uygulanıyor...")
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()

        return tokenizer, model
