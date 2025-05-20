import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def cosine_similarity(e1, e2):
    if len(e1) != len(e2):
        raise ValueError("Embeddings must have the same dimension")
    dot_product = sum(a * b for a, b in zip(e1, e2))
    norm1 = math.sqrt(sum(a * a for a in e1))
    norm2 = math.sqrt(sum(b * b for b in e2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


class EmbeddingDistribution:
    def __init__(
        self,
        alpha=0.4,
        gamma_similarity=0.8,
        beta=4,
        max_embeddings=100,
        min_samples_before_removal=10,
    ):
        # Core embedding data
        self.embeddings = []
        self.preferences = []
        self.rewards_received = []
        # Holistic reward information
        self.reward_average = 0
        self.total_rewards_received = 0
        # Tuning parameters
        self.alpha = alpha
        self.gamma_similarity = gamma_similarity
        self.beta = beta
        self.max_embeddings = max_embeddings
        self.min_samples_before_removal = min_samples_before_removal

    def update_reward_average(self, new_reward):
        self.total_rewards_received += 1
        self.reward_average += (
            new_reward - self.reward_average
        ) / self.total_rewards_received

    def update_preferences_with_decay(self, decay_factor=0.95):
        """Apply time decay to all preferences periodically"""
        for i in range(len(self.preferences)):
            self.preferences[i] *= decay_factor

    def calculate_action_probability(self, embedding_index):
        # Simple softmax calculation
        max_z = max(self.preferences)
        exp_values = [math.exp(val - max_z) for val in self.preferences]
        sum_exp = sum(exp_values)
        return exp_values[embedding_index] / sum_exp

    def update_embedding(self, embedding_index, reward):
        if embedding_index < 0 or embedding_index >= len(self.embeddings):
            return
        for i in range(len(self.embeddings)):
            old_preference_val = self.preferences[i]
            new_preference_val = 0
            if i == embedding_index:
                new_preference_val = old_preference_val + self.alpha * (
                    reward - self.reward_average
                ) * (1 - self.calculate_action_probability(i))
                self.rewards_received[i] += 1
            else:
                new_preference_val = old_preference_val - self.alpha * (
                    reward - self.reward_average
                ) * self.calculate_action_probability(i)
            self.preferences[i] = new_preference_val
        self.update_reward_average(reward)

    def calculate_embedding_preference_with_ucb(self, index):
        if index > len(self.embeddings):
            return None
        return (
            self.beta
            * (math.log(self.total_rewards_received) / self.rewards_received[index])
            + self.preferences[index]
        )

    def remove_with_protection_and_confidence(self):
        """
        Combined approach:
        1. Protects new embeddings until they've been sampled enough times
        2. Uses confidence bounds for well-explored embeddings
        """
        min_value = float("inf")
        min_value_index = -1
        mature_embeddings_exist = False

        # PHASE 1: Try to find mature embeddings (those with enough samples)
        # and use lower confidence bounds for them
        for i in range(len(self.embeddings)):
            if self.rewards_received[i] >= self.min_samples_before_removal:
                mature_embeddings_exist = True
                # Calculate lower confidence bound for mature embeddings
                uncertainty = self.beta * math.sqrt(1.0 / self.rewards_received[i])
                lower_bound = self.preferences[i] - uncertainty

                if lower_bound < min_value:
                    min_value = lower_bound
                    min_value_index = i

        # PHASE 2: If no mature embeddings exist, find the worst-performing
        # immature embedding based on preference only
        if not mature_embeddings_exist:
            for i in range(len(self.embeddings)):
                # For immature embeddings, just use preference value
                if self.preferences[i] < min_value:
                    min_value = self.preferences[i]
                    min_value_index = i

        # PHASE 3: Remove the selected embedding
        if min_value_index != -1:
            del self.embeddings[min_value_index]
            del self.preferences[min_value_index]
            del self.rewards_received[min_value_index]
        else:
            # This should never happen as long as there's at least one embedding
            raise ValueError("No embedding found to remove")

    def add_embedding(self, embedding, reward):
        if len(self.embeddings) > self.max_embeddings:
            self.remove_with_protection_and_confidence()
        self.embeddings.append(embedding)
        self.preferences.append(0)
        self.rewards_received.append(0)
        self.update_embedding(len(self.embeddings) - 1, reward)

    def handle_embedding(self, embedding, reward):
        self.update_preferences_with_decay()
        for i in range(len(self.embeddings)):
            curr_embedding = self.embeddings[i]
            if cosine_similarity(embedding, curr_embedding) >= self.gamma_similarity:
                # Treat embedding as a part of existing grouping and only do an update
                self.update_embedding(i, reward)
                return
        # If we reach here, there is no appropriate grouping so we create a new one
        self.add_embedding(embedding, reward)

    def recommend_embedding(
        self, temperature=1.0, target_exploration_prob=0.3, embedding_dim=384
    ):
        # If no embeddings exist, generate a random unit vector
        if not self.embeddings:
            # Generate random values from standard normal distribution
            random_embedding = np.random.normal(0, 1, embedding_dim)
            # Normalize to unit length
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
            return (
                random_embedding.tolist(),
                -1,
            )  # -1 indicates this is a generated vector

        # Determine whether to explore or exploit based on target_exploration_prob
        if np.random.random() < target_exploration_prob:
            # Exploration: Generate a random unit vector
            embedding_dim = len(self.embeddings[0])
            # Generate random values from standard normal distribution
            random_embedding = np.random.normal(0, 1, embedding_dim)
            # Normalize to unit length
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm

            return (
                random_embedding.tolist(),
                -1,
            )  # -1 indicates this is a generated vector
        else:
            # Exploitation: Use probability distribution based on preferences + ucb
            preferences_with_ucb = [
                self.calculate_embedding_preference_with_ucb(i)
                for i in range(len(self.preferences))
            ]
            max_z = max(preferences_with_ucb)
            exp_values = [
                math.exp((val - max_z) / temperature) for val in preferences_with_ucb
            ]
            sum_exp = sum(exp_values)
            probabilities = [exp_val / sum_exp for exp_val in exp_values]

            # Sample an index based on the probability distribution
            chosen_index = np.random.choice(len(self.embeddings), p=probabilities)
            chosen_embedding = self.embeddings[chosen_index].copy()

            return chosen_embedding, chosen_index

    def visualize_embedding_map(
        self,
        method="pca",
        save_path=None,
        figsize=(12, 10),
        min_point_size=50,
        max_point_size=500,
        colormap="viridis",
        show_indices=True,
        title=None,
    ):
        """
        Visualize embeddings on a 2D map with size representing preference value.

        Parameters:
        -----------
        method : str, optional (default='pca')
            Dimensionality reduction method to use. Options: 'pca', 'tsne'
        save_path : str, optional
            If provided, saves the figure to this path
        figsize : tuple, optional (default=(12, 10))
            Figure size
        min_point_size : int, optional (default=50)
            Minimum size for points on the plot
        max_point_size : int, optional (default=500)
            Maximum size for points on the plot
        colormap : str, optional (default='viridis')
            Matplotlib colormap to use for points
        show_indices : bool, optional (default=True)
            Whether to show embedding indices next to points
        title : str, optional
            Custom title for the plot. If None, a default title is generated.

        Returns:
        --------
        fig, ax : matplotlib Figure and Axes objects
        """
        if not self.embeddings:
            print("No embeddings to visualize.")
            return None, None

        # Convert embeddings to numpy array for dimensionality reduction
        embeddings_array = np.array(self.embeddings)

        # Perform dimensionality reduction
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            method_name = "PCA"
        elif method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(self.embeddings) - 1),
            )
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            method_name = "t-SNE"
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Scale preferences to point sizes
        # Handle negative preferences by shifting to ensure all values are positive
        preferences = np.array(self.preferences)
        min_pref = np.min(preferences)

        # Always shift preferences to make all values >= 0
        shifted_preferences = (
            preferences - min_pref + 0.1
        )  # Add a small constant to avoid zeros

        # If all preferences are the same, use a fixed size
        if np.max(shifted_preferences) == np.min(shifted_preferences):
            sizes = (
                np.ones_like(shifted_preferences)
                * (min_point_size + max_point_size)
                / 2
            )
        else:
            # Scale preferences to the range [min_point_size, max_point_size]
            sizes = min_point_size + (
                shifted_preferences - np.min(shifted_preferences)
            ) / (np.max(shifted_preferences) - np.min(shifted_preferences)) * (
                max_point_size - min_point_size
            )

        # Use reward counts for color intensity
        rewards_received = np.array(self.rewards_received)
        if np.max(rewards_received) == np.min(rewards_received):
            colors = np.ones_like(rewards_received) * 0.5
        else:
            colors = (rewards_received - np.min(rewards_received)) / (
                np.max(rewards_received) - np.min(rewards_received)
            )

        # Create scatter plot
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            s=sizes,
            c=colors,
            cmap=colormap,
            alpha=0.7,
        )

        # Add colorbar to represent reward counts
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Number of Rewards Received")

        # Add embedding indices
        if show_indices:
            for i, (x, y) in enumerate(reduced_embeddings):
                ax.annotate(str(i), (x, y), fontsize=8, ha="center", va="center")

        # Add title and labels
        if title is None:
            title = f"Embedding Map ({method_name} Projection)"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)

        # Add legend for point size
        if np.min(preferences) != np.max(preferences):
            # Create dummy scatter points for legend
            min_pref_actual = np.min(self.preferences)
            max_pref_actual = np.max(self.preferences)
            mid_pref_actual = (min_pref_actual + max_pref_actual) / 2

            # Add legend handles manually
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Min Preference: {min_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt(min_point_size) / 2,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Mid Preference: {mid_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt((min_point_size + max_point_size) / 4),
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Max Preference: {max_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt(max_point_size) / 2,
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        # Add information about number of embeddings and similarity threshold
        info_text = (
            f"Total Embeddings: {len(self.embeddings)}\n"
            f"Similarity Threshold: {self.gamma_similarity:.2f}\n"
            f"Min Samples Before Removal: {self.min_samples_before_removal}"
        )

        ax.text(
            0.02,
            0.02,
            info_text,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            fontsize=10,
        )

        plt.tight_layout()

        # Save figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def create_interactive_visualization(self, method="pca"):
        """
        Creates an interactive visualization of embeddings using Plotly.
        This provides more interactivity than the static matplotlib version.

        Parameters:
        -----------
        method : str, optional (default='pca')
            Dimensionality reduction method to use. Options: 'pca', 'tsne'

        Returns:
        --------
        fig : plotly.graph_objects.Figure
            An interactive Plotly figure
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import pandas as pd
        except ImportError:
            print(
                "Plotly, pandas, and sklearn are required for interactive visualization."
            )
            print("Install with: pip install plotly pandas scikit-learn")
            return None

        if not self.embeddings:
            print("No embeddings to visualize.")
            return None

        # Convert embeddings to numpy array for dimensionality reduction
        embeddings_array = np.array(self.embeddings)

        # Perform dimensionality reduction
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            method_name = "PCA"
        elif method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(self.embeddings) - 1),
            )
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            method_name = "t-SNE"
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

        # Scale preferences for marker size - FIXING THE NEGATIVE VALUES ISSUE
        preferences = np.array(self.preferences)
        min_pref = np.min(preferences)

        # Always shift preferences to ensure positive values for sizes
        shifted_preferences = (
            preferences - min_pref + 1
        )  # Add 1 to ensure all values are positive

        # Create a DataFrame for plotly
        df = pd.DataFrame(
            {
                "x": reduced_embeddings[:, 0],
                "y": reduced_embeddings[:, 1],
                "preference": self.preferences,  # Original preference for display
                "size_value": shifted_preferences,  # Shifted preference for size
                "rewards_received": self.rewards_received,
                "index": range(len(self.embeddings)),
            }
        )

        # Create interactive plot - using the SHIFTED values for size
        fig = px.scatter(
            df,
            x="x",
            y="y",
            size="size_value",  # Use the shifted values for size
            color="rewards_received",
            hover_data=[
                "index",
                "preference",
                "rewards_received",
            ],  # Show original preference in hover
            labels={
                "x": f"{method_name} Dimension 1",
                "y": f"{method_name} Dimension 2",
                "preference": "Preference",
                "size_value": "Size Value (Shifted Preference)",
                "rewards_received": "Rewards Received",
            },
            title=f"Interactive Embedding Map ({method_name} Projection)",
        )

        # Customize layout
        fig.update_layout(
            coloraxis_colorbar=dict(title="Rewards Received"), width=900, height=700
        )

        # Add text annotations for indices
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row["x"], y=row["y"], text=str(i), showarrow=False, font=dict(size=10)
            )

        # Add a note about preference scaling
        fig.add_annotation(
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            text=f"Note: Point sizes based on shifted preference values<br>Original range: [{min_pref:.2f}, {max(preferences):.2f}]",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
        )

        return fig
