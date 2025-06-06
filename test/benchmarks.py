from src.recommender import Recommender
import pandas as pd
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    def __init__(
        self, dataset_path, alpha, gamma, beta, max_embeddings, exp_rate, n_components
    ):
        print("Loading recommender...")
        self.recommender = Recommender(
            dataset_path, alpha, gamma, beta, max_embeddings, 10, n_components
        )
        self.exp_rate = exp_rate
        print("Loading file categories...")
        self.file_to_cat = self.load_file_and_categories(dataset_path)

    def load_file_and_categories(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Check if required columns exist
            if "filename" not in df.columns:
                raise KeyError("Column 'filename' not found in CSV")
            if "category" not in df.columns:
                raise KeyError("Column 'category' not found in CSV")
            # Create dictionary mapping filename to category
            filename_category_dict = dict(zip(df["filename"], df["category"]))
            return filename_category_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")

    def generate_reward(self, preferences, filename, reward_max):
        category = self.file_to_cat[int(filename)]
        preference_probability = preferences[category]
        return np.random.choice(
            [-reward_max, reward_max],
            p=[1 - preference_probability, preference_probability],
        )

    def run_benchmark(self, preferences, num_runs, reward_max):
        if set(preferences.keys()) != set(categories):
            raise RuntimeError("Provided preferences have invalid keys")
        print("Running benchmark...")
        result_preferences = {category: 0 for category in categories}
        rewards_history = []
        category_history = []

        for i in range(num_runs):
            file, embedding = self.recommender.recommend_article(self.exp_rate)
            file_cat = self.file_to_cat[int(file)]
            result_preferences[file_cat] += 1
            category_history.append(file_cat)

            reward = self.generate_reward(preferences, file, reward_max)
            rewards_history.append(reward)
            self.recommender.register_reaction(embedding, reward)

            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_runs} runs completed")

        return result_preferences, rewards_history, category_history


class ParameterTester:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.results = []

    def test_single_configuration(self, params, preferences, num_runs, reward_max):
        """Test a single parameter configuration"""
        benchmark = Benchmark(
            self.dataset_path,
            params["alpha"],
            params["gamma"],
            params["beta"],
            params["max_embeddings"],
            params["exp_rate"],
            params["n_components"],
        )

        result_preferences, rewards_history, category_history = benchmark.run_benchmark(
            preferences, num_runs, reward_max
        )

        # Calculate metrics
        total_reward = sum(rewards_history)
        avg_reward = np.mean(rewards_history)
        reward_std = np.std(rewards_history)

        # Calculate preference alignment score
        preference_alignment = self.calculate_preference_alignment(
            preferences, result_preferences, num_runs
        )

        # Calculate diversity score
        diversity_score = self.calculate_diversity_score(result_preferences)

        return {
            "params": params,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "preference_alignment": preference_alignment,
            "diversity_score": diversity_score,
            "result_preferences": result_preferences,
            "rewards_history": rewards_history,
            "category_history": category_history,
        }

    def calculate_preference_alignment(
        self, true_preferences, observed_counts, total_runs
    ):
        """Calculate how well the recommendations align with true preferences"""
        alignment_score = 0
        for category, true_pref in true_preferences.items():
            observed_pref = observed_counts[category] / total_runs
            # Use squared error
            alignment_score += (true_pref - observed_pref) ** 2
        # Return negative score (lower is better)
        return -alignment_score

    def calculate_diversity_score(self, result_preferences):
        """Calculate diversity of recommendations using entropy"""
        total = sum(result_preferences.values())
        if total == 0:
            return 0
        probs = [count / total for count in result_preferences.values() if count > 0]
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy

    def grid_search(self, param_grid, preferences_dict, num_runs=1000, reward_max=10):
        """Perform grid search over parameter combinations"""
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"Testing {len(all_combinations)} parameter combinations...")

        for pref_name, preferences in preferences_dict.items():
            print(f"\nTesting with {pref_name} preferences...")

            for combination in tqdm(all_combinations, desc=f"{pref_name} progress"):
                params = dict(zip(param_names, combination))

                try:
                    result = self.test_single_configuration(
                        params, preferences, num_runs, reward_max
                    )
                    result["preference_type"] = pref_name
                    self.results.append(result)
                except Exception as e:
                    print(f"Error with params {params}: {e}")

    def save_results(self, filename="benchmark_results.json"):
        """Save results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "params": result["params"],
                "preference_type": result["preference_type"],
                "total_reward": float(result["total_reward"]),
                "avg_reward": float(result["avg_reward"]),
                "reward_std": float(result["reward_std"]),
                "preference_alignment": float(result["preference_alignment"]),
                "diversity_score": float(result["diversity_score"]),
                "result_preferences": result["result_preferences"],
            }
            serializable_results.append(serializable_result)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")

    def load_results(self, filename="benchmark_results.json"):
        """Load results from JSON file"""
        with open(filename, "r") as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} results from {filename}")

    def analyze_results(self):
        """Analyze and visualize the results"""
        if not self.results:
            print("No results to analyze!")
            return

        # Convert to DataFrame for easier analysis
        df_data = []
        for result in self.results:
            row = result["params"].copy()
            row["preference_type"] = result["preference_type"]
            row["avg_reward"] = result["avg_reward"]
            row["preference_alignment"] = result["preference_alignment"]
            row["diversity_score"] = result["diversity_score"]
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Find best configurations for each preference type
        print("\nBest configurations by average reward:")
        for pref_type in df["preference_type"].unique():
            pref_df = df[df["preference_type"] == pref_type]
            best_idx = pref_df["avg_reward"].idxmax()
            best_config = pref_df.loc[best_idx]
            print(f"\n{pref_type}:")
            print(best_config)

        # Create visualizations
        self.create_visualizations(df)

        return df

    def create_visualizations(self, df):
        """Create various visualizations of the results"""
        # Set up the plotting style
        plt.style.use("seaborn-v0_8-darkgrid")

        # 1. Parameter importance heatmap
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        params_to_plot = [
            "alpha",
            "gamma",
            "beta",
            "max_embeddings",
            "exp_rate",
            "n_components",
        ]

        for i, param in enumerate(params_to_plot):
            param_avg = df.groupby(param)["avg_reward"].mean().reset_index()
            axes[i].plot(param_avg[param], param_avg["avg_reward"], marker="o")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Average Reward")
            axes[i].set_title(f"Impact of {param} on Performance")

        plt.tight_layout()
        plt.savefig("parameter_impact.png")
        plt.close()

        # 2. Preference type comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column="avg_reward", by="preference_type", ax=ax)
        ax.set_title("Performance Across Different Preference Types")
        ax.set_xlabel("Preference Type")
        ax.set_ylabel("Average Reward")
        plt.savefig("preference_comparison.png")
        plt.close()

        # 3. Correlation heatmap
        numeric_cols = [
            "alpha",
            "gamma",
            "beta",
            "max_embeddings",
            "exp_rate",
            "n_components",
            "avg_reward",
            "preference_alignment",
            "diversity_score",
        ]
        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Parameter Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

        print("\nVisualizations saved!")


# Parameter grids for testing
param_grids = {
    "quick_test": {
        "alpha": [0.2, 0.4, 0.6],
        "gamma": [0.7, 0.85, 0.95],
        "beta": [4, 8],
        "max_embeddings": [100],
        "exp_rate": [0.2],
        "n_components": [128, 256],
    },
    "full_test": {
        "alpha": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "gamma": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98],
        "beta": [2, 4, 6, 8],
        "max_embeddings": [25, 50, 75, 100],
        "exp_rate": [0.1, 0.15, 0.2, 0.25, 0.3],
        "n_components": [10, 25, 50, 100, 128, 192, 256],
    },
}

# Preference profiles
computer_science_preference = {
    "alt-atheism": 0.05,
    "comp-graphics": 0.8,
    "comp-os-ms-windows-misc": 0.8,
    "comp-sys-ibm-pc-hardware": 0.8,
    "comp-sys-mac-hardware": 0.9,
    "comp-windows-x": 0.8,
    "misc-forsale": 0.2,
    "rec-autos": 0.2,
    "rec-motorcycles": 0.2,
    "rec-sport-baseball": 0.2,
    "rec-sport-hockey": 0.2,
    "sci-crypt": 0.6,
    "sci-electronics": 0.5,
    "sci-med": 0.45,
    "sci-space": 0.5,
    "soc-religion-christian": 0.1,
    "talk-politics-guns": 0.15,
    "talk-politics-mideast": 0.1,
    "talk-politics-misc": 0.2,
    "talk-religion-misc": 0.1,
}

sports_preference = {
    "alt-atheism": 0.05,
    "comp-graphics": 0.1,
    "comp-os-ms-windows-misc": 0.15,
    "comp-sys-ibm-pc-hardware": 0.2,
    "comp-sys-mac-hardware": 0.2,
    "comp-windows-x": 0.1,
    "misc-forsale": 0.3,
    "rec-autos": 0.4,
    "rec-motorcycles": 0.5,
    "rec-sport-baseball": 0.8,
    "rec-sport-hockey": 0.9,
    "sci-crypt": 0.2,
    "sci-electronics": 0.1,
    "sci-med": 0.05,
    "sci-space": 0.1,
    "soc-religion-christian": 0.05,
    "talk-politics-guns": 0.3,
    "talk-politics-mideast": 0.1,
    "talk-politics-misc": 0.1,
    "talk-religion-misc": 0.05,
}

religion_preference = {
    "alt-atheism": 0.8,
    "comp-graphics": 0.1,
    "comp-os-ms-windows-misc": 0.15,
    "comp-sys-ibm-pc-hardware": 0.2,
    "comp-sys-mac-hardware": 0.2,
    "comp-windows-x": 0.1,
    "misc-forsale": 0.2,
    "rec-autos": 0.1,
    "rec-motorcycles": 0.1,
    "rec-sport-baseball": 0.2,
    "rec-sport-hockey": 0.1,
    "sci-crypt": 0.2,
    "sci-electronics": 0.1,
    "sci-med": 0.05,
    "sci-space": 0.1,
    "soc-religion-christian": 0.8,
    "talk-politics-guns": 0.3,
    "talk-politics-mideast": 0.6,
    "talk-politics-misc": 0.3,
    "talk-religion-misc": 0.9,
}

# Balanced preference (for testing)
balanced_preference = {category: 0.5 for category in categories}

# Main execution
if __name__ == "__main__":
    # Configuration
    dataset_path = "data/combined_shuffled.csv"  # Update this path
    num_runs = 500  # Number of recommendations per test
    reward_max = 10

    # Initialize tester
    tester = ParameterTester(dataset_path)

    # Define preference profiles to test
    preferences_to_test = {
        "computer_science": computer_science_preference,
        "sports": sports_preference,
        "religion": religion_preference,
        "balanced": balanced_preference,
    }

    # Run quick test
    print("Starting parameter testing...")
    tester.grid_search(
        param_grids["quick_test"],  # Use 'full_test' for comprehensive testing
        preferences_to_test,
        num_runs=num_runs,
        reward_max=reward_max,
    )

    # Save results
    tester.save_results("benchmark_results.json")

    # Analyze results
    results_df = tester.analyze_results()
