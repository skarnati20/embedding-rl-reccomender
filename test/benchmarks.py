from src.recommender import Recommender
import pandas as pd
import numpy as np

alpha_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma_vals = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
beta_vals = [2, 4, 6, 8]
max_embeddings = [25, 50, 75, 100]
exploration_rates = [0.1, 0.15, 0.2, 0.25, 0.3]
dimensionalities = [10, 25, 50, 100, 128, 192, 256]
reward_max = [5, 10, 20, 30, 40, 50]

categories = [
    "alt-atheism",
    "comp-graphics",
    "comp-os-ms-windows-misc",
    "comp-sys-ibm-pc-hardware",
    "comp-sys-mac-hardware",
    "comp-windows-x",
    "misc-forsale",
    "rec-autos",
    "rec-motorcycles",
    "rec-sport-baseball",
    "rec-sport-hockey",
    "sci-crypt",
    "sci-electronics",
    "sci-med",
    "sci-space",
    "soc-religion-christian",
    "talk-politics-guns",
    "talk-politics-mideast",
    "talk-politics-misc",
    "talk-religion-misc",
]

class Benchmark:
    def __init__(self, dataset_path, alpha, gamma, beta, max_embeddings, exp_rate, n_components):
        print("Loading recommender...")
        self.recommender = Recommender(dataset_path, alpha, gamma, beta, max_embeddings, 10, n_components)
        self.exp_rate = exp_rate
        print("Loading file categories...")
        self.file_to_cat = self.load_file_and_categories(dataset_path)
        
    def load_file_and_categories(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Check if required columns exist
            if 'filename' not in df.columns:
                raise KeyError("Column 'filename' not found in CSV")
            if 'category' not in df.columns:
                raise KeyError("Column 'category' not found in CSV")
            # Create dictionary mapping filename to category
            filename_category_dict = dict(zip(df['filename'], df['category']))
            return filename_category_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
    def generate_reward(self, preferences, filename, reward_max):
        category = self.file_to_cat[int(filename)]
        preference_probability = preferences[category]
        return np.random.choice([-reward_max, reward_max], p=[1 - preference_probability, preference_probability])
        
    def run_benchmark(self, preferences, num_runs, reward_max):
        if set(preferences.keys()) != set(categories):
            raise RuntimeError("Provided preferences have invalid keys")
        print("Running benchmark...")
        result_preferences = {category : 0 for category in categories}
        for _ in range(num_runs):
            file, embedding = self.recommender.recommend_article(self.exp_rate)
            file_cat = self.file_to_cat[int(file)]
            result_preferences[file_cat] += 1
            print(result_preferences)
            reward = self.generate_reward(preferences, file, reward_max)
            self.recommender.register_reaction(embedding, reward)
        self.recommender.visualize_embedding_map()
            
benchmark = Benchmark("data/combined_shuffled.csv", 0.4, 0.7, 4, 100, 0.2, 128)
estimated_preference = {
    "alt-atheism" : 0.2,
    "comp-graphics" : 0.8, 
    "comp-os-ms-windows-misc" : 0.7,
    "comp-sys-ibm-pc-hardware" : 0.8,
    "comp-sys-mac-hardware" : 0.9,
    "comp-windows-x" : 0.8,
    "misc-forsale" : 0.2,
    "rec-autos" : 0.2,
    "rec-motorcycles" : 0.2,
    "rec-sport-baseball" : 0.2,
    "rec-sport-hockey" : 0.2,
    "sci-crypt" : 0.2,
    "sci-electronics" : 0.2,
    "sci-med" : 0.2,
    "sci-space" : 0.2,
    "soc-religion-christian" : 0.2,
    "talk-politics-guns" : 0.3,
    "talk-politics-mideast" : 0.4,
    "talk-politics-misc" : 0.3,
    "talk-religion-misc" : 0.2,
}
benchmark.run_benchmark(estimated_preference, 2000, 20)