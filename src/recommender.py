import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from embedding_distribution import EmbeddingDistribution


class Recommender:
    """
    A recommendation system that uses embeddings to suggest articles/content based on user preferences.
    
    The recommender loads a dataset of articles with their embeddings, reduces dimensionality for efficiency,
    and uses an EmbeddingDistribution to learn user preferences through feedback. It can recommend new
    articles by finding the closest match to the user's learned preferences.
    """
    
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
        """
        Initialize the Recommender with a dataset of articles and their embeddings.
        
        This constructor:
        1. Loads the dataset from CSV (expecting 'filename' and 'embedding' columns)
        2. Reduces embedding dimensionality using Incremental PCA for efficiency
        3. Sets up an EmbeddingDistribution to track user preferences
        
        Args:
            dataset_path (str): Path to CSV file containing filename and embedding columns
            alpha (float): Learning rate for preference updates in EmbeddingDistribution
            gamma_similarity (float): Cosine similarity threshold for grouping embeddings
            beta (float): Exploration coefficient for UCB calculations
            max_embeddings (int): Maximum number of embeddings to store in user profile
            min_samples_before_removal (int): Minimum samples before an embedding can be removed
            n_components (int): Target dimensionality after PCA reduction
        """
        # Load the dataset from CSV
        df = pd.read_csv(dataset_path)

        embeddings_list = []
        filenames = []

        # Extract embeddings and filenames from the dataframe
        for _, row in df.iterrows():
            filename = str(row["filename"])
            embedding = row["embedding"]
            
            # Handle case where embedding is stored as string representation
            if isinstance(embedding, str):
                embedding = eval(embedding)  # Convert string representation to list/array
                
            embeddings_list.append(embedding)
            filenames.append(filename)

        # Convert to numpy array for efficient processing
        embeddings_array = np.array(embeddings_list)

        # Perform dimensionality reduction for computational efficiency
        print(
            f"Reducing dimensionality from {embeddings_array.shape[1]} to {n_components} dimensions..."
        )

        # Standardize embeddings to have zero mean and unit variance
        # This ensures all dimensions contribute equally to PCA
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings_array)

        # Use Incremental PCA for dimensionality reduction
        # Incremental PCA is preferred over regular PCA when we expect to process
        # data in batches or when the dataset is too large to fit in memory
        ipca = IncrementalPCA(n_components=n_components)
        ipca.fit(scaled_embeddings)
        reduced_embeddings = ipca.transform(scaled_embeddings)

        # Store the reduced embeddings in a dictionary mapping filename to embedding
        self.dataset = {}
        for i, filename in enumerate(filenames):
            self.dataset[filename] = reduced_embeddings[i]

        # Store the embedding dimension for future reference
        self.embedding_dimension = n_components
        
        # Initialize the user profile using EmbeddingDistribution
        # This will learn and track user preferences over time
        self.user_profile = EmbeddingDistribution(
            alpha=alpha,
            gamma_similarity=gamma_similarity,
            beta=beta,
            max_embeddings=max_embeddings,
            min_samples_before_removal=min_samples_before_removal,
        )

    def register_reaction(self, embedding, reaction):
        """
        Record a user's reaction to a specific embedding.
        
        This method feeds user feedback into the EmbeddingDistribution to update
        preferences. Positive reactions increase preference for similar embeddings,
        while negative reactions decrease it.
        
        Args:
            embedding: The embedding vector that the user reacted to
            reaction (float): The user's reaction score (positive = liked, negative = disliked)
        """
        self.user_profile.handle_embedding(embedding, reaction)

    def recommend_article(self, exploration_rate):
        """
        Recommend an article based on learned user preferences.
        
        This method:
        1. Gets a recommended embedding from the user profile (balancing exploration/exploitation)
        2. Finds the closest article in the dataset to that embedding
        3. Removes the recommended article from the dataset to avoid repeated recommendations
        
        Args:
            exploration_rate (float): Probability of exploration vs exploitation (0-1)
                                    Higher values = more exploration of new content
                                    Lower values = more exploitation of known preferences
        
        Returns:
            tuple: (filename, embedding) of the recommended article
            list: Empty list if no articles available
        """
        # Check if we have any articles left to recommend
        if not len(self.dataset):
            return []

        # Get a recommended embedding from the user profile
        # This balances between exploring new areas and exploiting known preferences
        recommended_embedding, _ = self.user_profile.recommend_embedding(
            self.embedding_dimension, 1, exploration_rate
        )
        
        if recommended_embedding is None:
            return []

        # Find the article in our dataset that's closest to the recommended embedding
        nearest_file, nearest_embedding = self.find_nearest_embedding(
            recommended_embedding
        )
        
        # Remove the recommended article from the dataset to avoid re-recommending it
        del self.dataset[nearest_file]
        
        return nearest_file, nearest_embedding

    def find_nearest_embedding(self, query_embedding):
        """
        Find the article with the embedding most similar to the query embedding.
        
        Uses cosine similarity to measure the angle between embeddings, which is
        effective for high-dimensional sparse data like text embeddings. Cosine
        similarity focuses on the direction rather than magnitude of vectors.
        
        Args:
            query_embedding: The target embedding to find matches for
            
        Returns:
            tuple: (filename, embedding) of the most similar article
            None: If no articles are available in the dataset
        """
        if not self.dataset:
            return None

        max_similarity = -1  # Cosine similarity ranges from -1 to 1
        nearest_file = None
        nearest_embedding = None

        # Iterate through all articles in the dataset
        for filename, embedding in self.dataset.items():
            # Calculate cosine similarity manually
            dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
            magnitude1 = math.sqrt(sum(a * a for a in query_embedding))
            magnitude2 = math.sqrt(sum(b * b for b in embedding))

            # Handle edge case where one of the embeddings is zero vector
            if magnitude1 == 0 or magnitude2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (magnitude1 * magnitude2)

            # Keep track of the most similar embedding found so far
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_file = filename
                nearest_embedding = embedding

        return nearest_file, nearest_embedding

    def visualize_embedding_map(self):
        """
        Create and display an interactive visualization of the user's embedding preferences.
        
        This method uses the EmbeddingDistribution's visualization capabilities to show:
        - The user's learned embedding preferences in 2D space
        - Point sizes representing preference strengths
        - Point colors representing how often embeddings have been sampled
        
        The visualization opens in a browser or displays in a Jupyter notebook,
        providing insights into what the user likes and how the recommendation
        system has learned their preferences.
        """
        fig = self.user_profile.create_interactive_visualization()
        fig.show()  # This will open in a browser or notebook