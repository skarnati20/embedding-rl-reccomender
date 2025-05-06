import pandas as pd
import numpy as np
import math
from embedding_distribution import EmbeddingDistribution


class Reccomender:
    def __init__(self, dataset_path):
        df = pd.read_csv(dataset_path)
        self.dataset = {}
        for _, row in df.iterrows():
            filename = str(row["filename"])
            embedding = row["embedding"]
            if isinstance(embedding, str):
                embedding = eval(embedding)
            self.dataset[filename] = embedding
        self.user_profile = EmbeddingDistribution()

    def register_reaction(self, embedding, reaction):
        self.user_profile.handle_embedding(embedding, reaction)

    def recommend_article(self):
        if not len(self.dataset):
            return []

        recommended_embedding, _ = self.user_profile.recommend_embedding()
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
