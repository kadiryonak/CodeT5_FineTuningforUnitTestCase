from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import subprocess

class Metrics:
    @staticmethod
    def calculate_bleu(reference, prediction):
        smoothie = SmoothingFunction().method4
        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=smoothie
        )

    @staticmethod
    def calculate_codebleu(reference_file, generated_file, lang='java'):
        # CodeBLEU hesaplama için subprocess ile script çağır
        command = f"python CodeXGLUE/CodeBLEU/calc_code_bleu.py --refs {reference_file} --hyps {generated_file} --lang {lang}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0

    @staticmethod
    def evaluate_code_metrics(test_dataset, model, tokenizer, lang='java'):

        references, hypotheses = [], []

        for item in test_dataset:
            reference_code = tokenizer.decode(item['labels'], skip_special_tokens=True)
            generated_code = tokenizer.decode(model.generate(item['input_ids'].unsqueeze(0)), skip_special_tokens=True)
            references.append(reference_code)
            hypotheses.append(generated_code)

        # BLEU hesapla
        bleu_scores = [
            Metrics.calculate_bleu(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]
        avg_bleu = np.mean(bleu_scores)

        # Dosyaları oluştur
        with open("reference.txt", "w") as ref_file, open("generated.txt", "w") as gen_file:
            for ref, hyp in zip(references, hypotheses):
                ref_file.write(ref + "\n")
                gen_file.write(hyp + "\n")

        # CodeBLEU hesapla
        avg_codebleu = Metrics.calculate_codebleu("reference.txt", "generated.txt", lang=lang)

        return {"BLEU": avg_bleu, "CodeBLEU": avg_codebleu}
