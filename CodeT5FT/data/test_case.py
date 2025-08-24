class TestCase:
    def __init__(self, reader):
        print("Test Case started")
        self.reader = reader
        self.features = {'train': [], 'eval': [], 'test': []}

    def analyze(self):
        print("Analyzing...")
        train, eval, test = self.reader.read_data()

        if not (train or eval or test):
            print("Error: No JSON data.")
            return

        self._extract_features(train, 'train')
        self._extract_features(eval, 'eval')
        self._extract_features(test, 'test')

        self._print_features()
        return self.features

    def _extract_features(self, data, dataset_name):
        for item in data:
            feature = {
                'target': item.get('target', 'Missing'),
                'focal_method': item.get('src_fm', 'Missing'),
                'focal_context_1': item.get('src_fm_fc', 'Missing'),
                'focal_context_2': item.get('src_fm_fc_co', 'Missing'),
                'focal_context_3': item.get('src_fm_fc_ms', 'Missing'),
                'focal_context_4': item.get('src_fm_fc_ms_ff', 'Missing'),
                'method_signature_length': item.get('src_fm', ''),
                'text_length': len(item.get('src_fm', '')),
                'keywords': self._extract_keywords(item.get('src_fm', ''))
            }
            self.features[dataset_name].append(feature)

    def _print_features(self):
        for dataset_name, features in self.features.items():
            print(f"\n{dataset_name.capitalize()} Data:")
            for feature in features:
                print(f"Target: {feature['target']}")
                print(f"Focal Method: {feature['focal_method']}")
                print(f"Method Signature Length: {feature['method_signature_length']}")
                print(f"Text Length: {feature['text_length']}")
                print('-' * 40)

    def _extract_keywords(self, text):
        if not text:
            return []
        return [word for word in text.split() if len(word) > 3]
