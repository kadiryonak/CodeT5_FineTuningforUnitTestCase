import os
import json
from abc import ABC, abstractmethod


class DataReader(ABC):
    @abstractmethod
    def read_data(self):
        pass


class JsonDataReader(DataReader):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train = []
        self.eval = []
        self.test = []

    def read_data(self):
        print("Dataset start to reading...")

        for root, dirs, files in os.walk(self.data_path):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)


                    if 'train' in root:
                        self._load_json(file_path, 'train')
                    elif 'eval' in root:
                        self._load_json(file_path, 'eval')
                    elif 'test' in root:
                        self._load_json(file_path, 'test')

  
        if not self.train and not self.eval and not self.test:
            print("Error: Json folder is not found.")

        return self.train, self.eval, self.test

    def _load_json(self, file_path, root):
        print(f"Loading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if 'train' in root:
                    self.train.append(data)
                elif 'eval' in root:
                    self.eval.append(data)
                elif 'test' in root:
                    self.test.append(data)
        except json.JSONDecodeError:
            print(f"Error: An error occurred while parsing the JSON file: {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
