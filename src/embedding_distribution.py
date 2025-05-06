import numpy as np
import math


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
    def __init__(self, alpha=0.2, gamma_similarity=0.75):
        self.embeddings = []
        self.preferences = []
        self.reward_average = 0
        self.rewards_received = 0
        self.alpha = alpha
        self.gamma_similarity = gamma_similarity

    def update_reward_average(self, new_reward):
        self.rewards_received += 1
        self.reward_average += (
            new_reward - self.reward_average
        ) / self.rewards_received

    def update_preferences_with_decay(self, decay_factor=0.99):
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
            else:
                new_preference_val = old_preference_val - self.alpha * (
                    reward - self.reward_average
                ) * self.calculate_action_probability(i)
            self.preferences[i] = new_preference_val
        self.update_reward_average(reward)

    def add_embedding(self, embedding, reward):
        self.embeddings.append(embedding)
        self.preferences.append(0)
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
            # Exploitation: Use probability distribution based on preferences
            max_z = max(self.preferences)
            exp_values = [
                math.exp((val - max_z) / temperature) for val in self.preferences
            ]
            sum_exp = sum(exp_values)
            probabilities = [exp_val / sum_exp for exp_val in exp_values]

            # Sample an index based on the probability distribution
            chosen_index = np.random.choice(len(self.embeddings), p=probabilities)
            chosen_embedding = self.embeddings[chosen_index].copy()

            return chosen_embedding, chosen_index
