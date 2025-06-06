import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def cosine_similarity(e1, e2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
        e1, e2: List or array-like embeddings to compare
        
    Returns:
        float: Cosine similarity value between -1 and 1
        
    Raises:
        ValueError: If embeddings have different dimensions
    """
    if len(e1) != len(e2):
        raise ValueError("Embeddings must have the same dimension")
    dot_product = sum(a * b for a, b in zip(e1, e2))
    norm1 = math.sqrt(sum(a * a for a in e1))
    norm2 = math.sqrt(sum(b * b for b in e2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


class EmbeddingDistribution:
    """
    A class that manages a collection of embeddings and their associated preferences
    using a multi-armed bandit approach with Upper Confidence Bound (UCB) exploration.
    
    The class groups similar embeddings together based on cosine similarity and
    maintains preference scores that are updated based on rewards received.
    """
    
    def __init__(
        self,
        alpha=0.4,
        gamma_similarity=0.8,
        beta=4,
        max_embeddings=100,
        min_samples_before_removal=10,
    ):
        """
        Initialize the EmbeddingDistribution with tuning parameters.
        
        Args:
            alpha (float): Learning rate for preference updates
            gamma_similarity (float): Cosine similarity threshold for grouping embeddings
            beta (float): Exploration coefficient for UCB calculations
            max_embeddings (int): Maximum number of embeddings to store
            min_samples_before_removal (int): Minimum samples before an embedding can be removed
        """
        # Core embedding data
        self.embeddings = []  # List of embedding vectors
        self.preferences = []  # Preference scores for each embedding
        self.rewards_received = []  # Count of rewards received per embedding
        
        # Holistic reward information
        self.reward_average = 0  # Running average of all rewards
        self.total_rewards_received = 0  # Total number of rewards processed
        
        # Tuning parameters
        self.alpha = alpha
        self.gamma_similarity = gamma_similarity
        self.beta = beta
        self.max_embeddings = max_embeddings
        self.min_samples_before_removal = min_samples_before_removal

    def update_reward_average(self, new_reward):
        """
        Update the running average of rewards using incremental formula.
        
        This maintains a global average without storing all historical rewards,
        which is used as a baseline for preference updates.
        
        Args:
            new_reward (float): The new reward value to incorporate
        """
        self.total_rewards_received += 1
        self.reward_average += (
            new_reward - self.reward_average
        ) / self.total_rewards_received

    def update_preferences_with_decay(self, decay_factor=0.95):
        """
        Apply time decay to all preference values to prevent old preferences
        from dominating and allow adaptation to changing reward patterns.
        
        Args:
            decay_factor (float): Multiplicative decay factor (0 < decay_factor < 1)
        """
        for i in range(len(self.preferences)):
            self.preferences[i] *= decay_factor

    def calculate_action_probability(self, embedding_index):
        """
        Calculate the probability of selecting a specific embedding using softmax.
        
        This converts preference values into a probability distribution,
        with higher preferences having higher selection probabilities.
        
        Args:
            embedding_index (int): Index of the embedding to calculate probability for
            
        Returns:
            float: Probability of selecting this embedding (0 to 1)
        """
        # Softmax calculation with numerical stability
        max_z = max(self.preferences)
        exp_values = [math.exp(val - max_z) for val in self.preferences]
        sum_exp = sum(exp_values)
        return exp_values[embedding_index] / sum_exp

    def update_embedding(self, embedding_index, reward):
        """
        Update preference values for all embeddings based on a reward received
        for a specific embedding. This implements a policy gradient-like update.
        
        The selected embedding gets a positive update proportional to how much
        better the reward was than average, while non-selected embeddings get
        negative updates to maintain probability mass conservation.
        
        Args:
            embedding_index (int): Index of the embedding that received the reward
            reward (float): The reward value received
        """
        if embedding_index < 0 or embedding_index >= len(self.embeddings):
            return
            
        for i in range(len(self.embeddings)):
            old_preference_val = self.preferences[i]
            
            if i == embedding_index:
                # Update for the selected embedding: increase preference if reward > average
                new_preference_val = old_preference_val + self.alpha * (
                    reward - self.reward_average
                ) * (1 - self.calculate_action_probability(i))
                self.rewards_received[i] += 1
            else:
                # Update for non-selected embeddings: decrease preference proportionally
                new_preference_val = old_preference_val - self.alpha * (
                    reward - self.reward_average
                ) * self.calculate_action_probability(i)
                
            self.preferences[i] = new_preference_val
            
        self.update_reward_average(reward)

    def calculate_embedding_preference_with_ucb(self, index):
        """
        Calculate the Upper Confidence Bound (UCB) value for an embedding.
        
        UCB balances exploitation (using current preference) with exploration
        (adding uncertainty bonus for less-sampled embeddings).
        
        Args:
            index (int): Index of the embedding
            
        Returns:
            float: UCB value combining preference and exploration bonus
            None: If index is invalid
        """
        if index > len(self.embeddings):
            return None
        return (
            self.beta
            * (math.log(self.total_rewards_received) / self.rewards_received[index])
            + self.preferences[index]
        )

    def remove_with_protection_and_confidence(self):
        """
        Remove the worst-performing embedding using a two-phase approach:
        
        Phase 1: For mature embeddings (with sufficient samples), use lower
                confidence bounds to identify truly poor performers
        Phase 2: If no mature embeddings exist, remove the worst immature one
                based on preference alone
        
        This protects new embeddings from premature removal while still
        maintaining the embedding limit.
        """
        min_value = float("inf")
        min_value_index = -1
        mature_embeddings_exist = False

        # PHASE 1: Try to find mature embeddings and use confidence bounds
        for i in range(len(self.embeddings)):
            if self.rewards_received[i] >= self.min_samples_before_removal:
                mature_embeddings_exist = True
                # Calculate lower confidence bound for mature embeddings
                uncertainty = self.beta * math.sqrt(1.0 / self.rewards_received[i])
                lower_bound = self.preferences[i] - uncertainty

                if lower_bound < min_value:
                    min_value = lower_bound
                    min_value_index = i

        # PHASE 2: If no mature embeddings exist, use preference values only
        if not mature_embeddings_exist:
            for i in range(len(self.embeddings)):
                if self.preferences[i] < min_value:
                    min_value = self.preferences[i]
                    min_value_index = i

        # PHASE 3: Remove the selected embedding
        if min_value_index != -1:
            del self.embeddings[min_value_index]
            del self.preferences[min_value_index]
            del self.rewards_received[min_value_index]
        else:
            raise ValueError("No embedding found to remove")

    def add_embedding(self, embedding, reward):
        """
        Add a new embedding to the collection, removing an old one if necessary.
        
        If we're at the maximum capacity, this method first removes the worst
        embedding before adding the new one. The new embedding starts with
        zero preference and gets its first update from the provided reward.
        
        Args:
            embedding: The embedding vector to add
            reward (float): Initial reward for this embedding
        """
        if len(self.embeddings) > self.max_embeddings:
            self.remove_with_protection_and_confidence()
            
        self.embeddings.append(embedding)
        self.preferences.append(0)  # Start with neutral preference
        self.rewards_received.append(0)  # No rewards received yet
        
        # Give the new embedding its first update
        self.update_embedding(len(self.embeddings) - 1, reward)

    def handle_embedding(self, embedding, reward):
        """
        Main method to process a new embedding and its associated reward.
        
        This method:
        1. Applies time decay to existing preferences
        2. Checks if the new embedding is similar to any existing ones
        3. If similar embedding found: updates its preference
        4. If no similar embedding: adds as new embedding
        
        Args:
            embedding: The embedding vector to process
            reward (float): The reward associated with this embedding
        """
        # Apply time decay to prevent stale preferences from dominating
        self.update_preferences_with_decay()
        
        # Check if this embedding is similar to any existing ones
        for i in range(len(self.embeddings)):
            curr_embedding = self.embeddings[i]
            if cosine_similarity(embedding, curr_embedding) >= self.gamma_similarity:
                # Found similar embedding - just update its preference
                self.update_embedding(i, reward)
                return
                
        # No similar embedding found - add as new embedding
        self.add_embedding(embedding, reward)

    def recommend_embedding(
        self, embedding_dim, temperature=1.0, target_exploration_prob=0.3
    ):
        """
        Recommend an embedding to use, balancing exploration and exploitation.
        
        Uses epsilon-greedy approach:
        - With probability target_exploration_prob: return random unit vector (explore)
        - Otherwise: sample from stored embeddings using UCB-weighted probabilities (exploit)
        
        Args:
            embedding_dim (int): Dimension of embeddings to generate if exploring
            temperature (float): Temperature parameter for softmax sampling
            target_exploration_prob (float): Probability of exploration vs exploitation
            
        Returns:
            tuple: (embedding_vector, index)
                   index = -1 indicates a randomly generated exploration vector
        """
        # If no embeddings exist, generate a random unit vector
        if not self.embeddings:
            random_embedding = np.random.normal(0, 1, embedding_dim)
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
            return (random_embedding.tolist(), -1)

        # Epsilon-greedy exploration vs exploitation
        if np.random.random() < target_exploration_prob:
            # EXPLORATION: Generate a random unit vector
            embedding_dim = len(self.embeddings[0])
            random_embedding = np.random.normal(0, 1, embedding_dim)
            norm = np.linalg.norm(random_embedding)
            if norm > 0:
                random_embedding = random_embedding / norm
            return (random_embedding.tolist(), -1)
        else:
            # EXPLOITATION: Sample from existing embeddings using UCB preferences
            preferences_with_ucb = [
                self.calculate_embedding_preference_with_ucb(i)
                for i in range(len(self.preferences))
            ]
            
            # Convert to probability distribution using softmax with temperature
            max_z = max(preferences_with_ucb)
            exp_values = [
                math.exp((val - max_z) / temperature) for val in preferences_with_ucb
            ]
            sum_exp = sum(exp_values)
            probabilities = [exp_val / sum_exp for exp_val in exp_values]

            # Sample an embedding based on the probability distribution
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
        Create a 2D visualization of embeddings using dimensionality reduction.
        
        This method projects high-dimensional embeddings to 2D space and creates
        a scatter plot where:
        - Point size represents preference value (larger = higher preference)
        - Point color represents number of rewards received (darker = more rewards)
        - Point position shows embedding relationships in reduced space
        
        Args:
            method (str): Dimensionality reduction method ('pca' or 'tsne')
            save_path (str): Optional path to save the figure
            figsize (tuple): Figure size in inches
            min_point_size (int): Minimum point size in the plot
            max_point_size (int): Maximum point size in the plot
            colormap (str): Matplotlib colormap for point colors
            show_indices (bool): Whether to show embedding indices as labels
            title (str): Custom title for the plot
            
        Returns:
            tuple: (fig, ax) matplotlib Figure and Axes objects
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

        # Scale preferences to point sizes, handling negative values
        preferences = np.array(self.preferences)
        min_pref = np.min(preferences)

        # Shift preferences to make all values positive for sizing
        shifted_preferences = preferences - min_pref + 0.1

        # Handle case where all preferences are equal
        if np.max(shifted_preferences) == np.min(shifted_preferences):
            sizes = np.ones_like(shifted_preferences) * (min_point_size + max_point_size) / 2
        else:
            # Scale to size range
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

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Number of Rewards Received")

        # Add embedding indices as labels
        if show_indices:
            for i, (x, y) in enumerate(reduced_embeddings):
                ax.annotate(str(i), (x, y), fontsize=8, ha="center", va="center")

        # Set title and labels
        if title is None:
            title = f"Embedding Map ({method_name} Projection)"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)

        # Add legend for point sizes
        if np.min(preferences) != np.max(preferences):
            min_pref_actual = np.min(self.preferences)
            max_pref_actual = np.max(self.preferences)
            mid_pref_actual = (min_pref_actual + max_pref_actual) / 2

            legend_elements = [
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    label=f"Min Preference: {min_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt(min_point_size) / 2,
                ),
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    label=f"Mid Preference: {mid_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt((min_point_size + max_point_size) / 4),
                ),
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    label=f"Max Preference: {max_pref_actual:.2f}",
                    markerfacecolor="gray",
                    markersize=np.sqrt(max_point_size) / 2,
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        # Add information text box
        info_text = (
            f"Total Embeddings: {len(self.embeddings)}\n"
            f"Similarity Threshold: {self.gamma_similarity:.2f}\n"
            f"Min Samples Before Removal: {self.min_samples_before_removal}"
        )

        ax.text(
            0.02, 0.02, info_text, transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7), fontsize=10,
        )

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def create_interactive_visualization(self, method="pca"):
        """
        Create an interactive visualization using Plotly instead of matplotlib.
        
        This provides the same information as visualize_embedding_map but with
        interactive features like zoom, pan, and hover information.
        
        Args:
            method (str): Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            plotly.graph_objects.Figure: Interactive Plotly figure, or None if dependencies missing
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import pandas as pd
        except ImportError:
            print("Plotly, pandas, and sklearn are required for interactive visualization.")
            print("Install with: pip install plotly pandas scikit-learn")
            return None

        if not self.embeddings:
            print("No embeddings to visualize.")
            return None

        # Convert embeddings to numpy array
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

        # Handle negative preferences for sizing
        preferences = np.array(self.preferences)
        min_pref = np.min(preferences)
        shifted_preferences = preferences - min_pref + 1  # Ensure positive values

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "preference": self.preferences,  # Original values for display
            "size_value": shifted_preferences,  # Shifted values for sizing
            "rewards_received": self.rewards_received,
            "index": range(len(self.embeddings)),
        })

        # Create interactive scatter plot
        fig = px.scatter(
            df, x="x", y="y",
            size="size_value",  # Use shifted values for sizing
            color="rewards_received",
            hover_data=["index", "preference", "rewards_received"],
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
            coloraxis_colorbar=dict(title="Rewards Received"),
            width=900, height=700
        )

        # Add index annotations
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row["x"], y=row["y"], text=str(i),
                showarrow=False, font=dict(size=10)
            )

        # Add preference scaling note
        fig.add_annotation(
            x=0.01, y=0.01, xref="paper", yref="paper",
            text=f"Note: Point sizes based on shifted preference values<br>"
                 f"Original range: [{min_pref:.2f}, {max(preferences):.2f}]",
            showarrow=False, font=dict(size=10), align="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray", borderwidth=1,
        )

        return fig