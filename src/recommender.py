import math
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from embedding_distribution import EmbeddingDistribution


class Recommender:
    """
    A recommendation system that uses embeddings to suggest articles/content based on user preferences.
    
    The recommender loads a dataset of articles with their embeddings, reduces dimensionality for efficiency,
    normalizes embeddings for cosine similarity, and uses Faiss for fast nearest neighbor search.
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
        3. Normalizes embeddings for cosine similarity calculations
        4. Sets up Faiss index for fast nearest neighbor search
        5. Sets up an EmbeddingDistribution to track user preferences
        
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
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        # Perform dimensionality reduction for computational efficiency
        print(
            f"Reducing dimensionality from {embeddings_array.shape[1]} to {n_components} dimensions..."
        )

        # Standardize embeddings to have zero mean and unit variance
        # This ensures all dimensions contribute equally to PCA
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings_array)

        # Use Incremental PCA for dimensionality reduction
        ipca = IncrementalPCA(n_components=n_components)
        ipca.fit(scaled_embeddings)
        reduced_embeddings = ipca.transform(scaled_embeddings).astype(np.float32)

        # Normalize embeddings for cosine similarity
        # After normalization, L2 distance = cosine similarity
        print("Normalizing embeddings for cosine similarity...")
        norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        self.normalized_embeddings = reduced_embeddings / norms

        # Store filenames and create mapping
        self.filenames = filenames
        self.filename_to_index = {filename: i for i, filename in enumerate(filenames)}
        
        # Store preprocessing components for future use
        self.scaler = scaler
        self.ipca = ipca
        self.embedding_dimension = n_components
        
        # Create Faiss index for fast similarity search
        print("Building Faiss index...")
        # Use IndexFlatL2 for exact search with L2 distance
        # On normalized vectors, L2 distance approximates cosine similarity
        self.faiss_index = faiss.IndexFlatL2(n_components)
        self.faiss_index.add(self.normalized_embeddings)
        
        # Keep track of available articles (for removal after recommendation)
        self.available_indices = set(range(len(filenames)))
        
        # Initialize the user profile using EmbeddingDistribution
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
        
        Args:
            embedding: The embedding vector that the user reacted to
            reaction (float): The user's reaction score (positive = liked, negative = disliked)
        """
        self.user_profile.handle_embedding(embedding, reaction)

    def recommend_article(self, exploration_rate, k_candidates=10):
        """
        Recommend an article based on learned user preferences using Faiss.
        
        Args:
            exploration_rate (float): Probability of exploration vs exploitation (0-1)
                                    Higher values = more exploration of new content
                                    Lower values = more exploitation of known preferences
        
        Returns:
            tuple: (filename, embedding) of the recommended article
            list: Empty list if no articles available
        """
        # Check if we have any articles left to recommend
        if not self.available_indices:
            return []

        # Get a recommended embedding from the user profile
        recommended_embedding, _ = self.user_profile.recommend_embedding(
            self.embedding_dimension, 1, exploration_rate
        )
        
        if recommended_embedding is None:
            return []

        # Find nearest neighbors using Faiss
        nearest_filename, nearest_embedding = self.find_nearest_embedding_faiss(
            recommended_embedding, k_candidates
        )
        
        if nearest_filename is None:
            return []
        
        # Remove the recommended article from available set
        article_index = self.filename_to_index[nearest_filename]
        self.available_indices.discard(article_index)
        
        return nearest_filename, nearest_embedding

    def find_nearest_embedding_faiss(self, query_embedding, k=10):
        """
        Find the most similar available article using Faiss.
        
        Args:
            query_embedding: The target embedding to find matches for
            k (int): Number of candidates to search through
            
        Returns:
            tuple: (filename, embedding) of the most similar available article
            None: If no articles are available
        """
        if not self.available_indices:
            return None, None

        # Reshape query for Faiss (needs to be 2D)
        query_reshaped = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search for k nearest neighbors
        # We search for more than we need in case some are already recommended
        search_k = min(k * 2, len(self.available_indices))
        distances, indices = self.faiss_index.search(query_reshaped, search_k)
        
        # Find the first available (not yet recommended) article
        for idx in indices[0]:
            if idx in self.available_indices:
                filename = self.filenames[idx]
                embedding = self.normalized_embeddings[idx]
                return filename, embedding
        
        # Fallback: if no available articles found in search results
        # (shouldn't happen with sufficient search_k)
        if self.available_indices:
            # Just return the first available article
            available_idx = next(iter(self.available_indices))
            filename = self.filenames[available_idx]
            embedding = self.normalized_embeddings[available_idx]
            return filename, embedding
        
        return None, None

    def find_nearest_embedding_manual(self, query_embedding):
        """
        Fallback method: Find the most similar embedding using manual cosine similarity.
        This is kept for compatibility but is slower than the Faiss method.
        
        Args:
            query_embedding: The target embedding to find matches for
            
        Returns:
            tuple: (filename, embedding) of the most similar available article
        """
        if not self.available_indices:
            return None, None

        max_similarity = -1
        nearest_filename = None
        nearest_embedding = None

        # Only check available articles
        for idx in self.available_indices:
            filename = self.filenames[idx]
            embedding = self.normalized_embeddings[idx]
            
            # For normalized vectors, dot product = cosine similarity
            similarity = np.dot(query_embedding, embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                nearest_filename = filename
                nearest_embedding = embedding

        return nearest_filename, nearest_embedding

    def visualize_embedding_map(self):
        """
        Create and display an interactive visualization of the user's embedding preferences.
        """
        fig = self.user_profile.create_interactive_visualization()
        fig.show()