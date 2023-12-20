import pandas as pd
import os


class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __str__(self):
        return f'{self.folder_path}'

    def load_and_process_data(self, fn):
        all_data = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.folder_path, filename)
                data = pd.read_csv(file_path)
                data['features'] = data['smiles'].apply(fn)
                all_data.append(data)
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
