import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from embedding_distribution import EmbeddingDistribution


class Recommender:
    def __init__(
        self,
        dataset_path,
        alpha=0.2,
        gamma_similarity=0.95,
        beta=2,
        max_embeddings=100,
        min_samples_before_removal=10,
        n_components=128,
    ):
        df = pd.read_csv(dataset_path)

        embeddings_list = []
        filenames = []

        for _, row in df.iterrows():
            filename = str(row["filename"])
            embedding = row["embedding"]
            if isinstance(embedding, str):
                embedding = eval(embedding)
            embeddings_list.append(embedding)
            filenames.append(filename)

        embeddings_array = np.array(embeddings_list)

        # Apply UMAP dimensionality reduction
        print(
            f"Reducing dimensionality from {embeddings_array.shape[1]} to {n_components} dimensions..."
        )

        # Standardize the embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings_array)

        # Perform Incremental PCA reduction
        ipca = IncrementalPCA(n_components=n_components)
        ipca.fit(scaled_embeddings)
        reduced_embeddings = ipca.transform(scaled_embeddings)

        # Store the reduced embeddings in the dataset dictionary
        self.dataset = {}
        for i, filename in enumerate(filenames):
            self.dataset[filename] = reduced_embeddings[i]

        self.embedding_dimension = n_components
        self.user_profile = EmbeddingDistribution(
            alpha=alpha,
            gamma_similarity=gamma_similarity,
            beta=beta,
            max_embeddings=max_embeddings,
            min_samples_before_removal=min_samples_before_removal,
        )

    def register_reaction(self, embedding, reaction):
        self.user_profile.handle_embedding(embedding, reaction)

    def recommend_article(self, exploration_rate):
        if not len(self.dataset):
            return []

        recommended_embedding, _ = self.user_profile.recommend_embedding(
            self.embedding_dimension, 1, exploration_rate
        )
        if recommended_embedding is None:
            return []

        nearest_file, nearest_embedding = self.find_nearest_embedding(
            recommended_embedding
        )
        del self.dataset[nearest_file]
        return nearest_file, nearest_embedding

    def find_nearest_embedding(self, query_embedding):
        """
        Find the filename with the embedding closest to the query_embedding
        using cosine similarity
        """
        if not self.dataset:
            return None

        max_similarity = -1
        nearest_file = None
        nearest_embedding = None

        for filename, embedding in self.dataset.items():
            dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
            magnitude1 = math.sqrt(sum(a * a for a in query_embedding))
            magnitude2 = math.sqrt(sum(b * b for b in embedding))

            if magnitude1 == 0 or magnitude2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (magnitude1 * magnitude2)

            if similarity > max_similarity:
                max_similarity = similarity
                nearest_file = filename
                nearest_embedding = embedding

        return nearest_file, nearest_embedding

    def visualize_embedding_map(self):
        fig = self.user_profile.create_interactive_visualization()
        fig.show()  # This will open in a browser or notebook
