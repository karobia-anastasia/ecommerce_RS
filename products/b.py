import pandas as pd
import numpy as np
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from products.models import Cart, Transaction, Product
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import logging
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from products.models import Cart, Transaction, Product
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Assume we have a function to get the user-item matrix
def get_user_item_matrix():
    """
    Creates a user-item matrix from the database.
    For simplicity, assume it's a pandas DataFrame with users as index, 
    products as columns, and ratings as values.
    """
    # Fetch user-product interactions (this can be from a Django model or any data source)
    interactions = UserProductInteraction.objects.all().values('user_id', 'product_id', 'rating')

    # Convert to pandas DataFrame
    df = pd.DataFrame(interactions)
    
    # Create a user-item matrix with users as index and products as columns
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    return user_item_matrix

def build_svd_model(user_item_matrix):
    """
    Builds the SVD model and similarity matrix.
    """
    # Perform Singular Value Decomposition (SVD) on the user-item matrix
    svd = TruncatedSVD(n_components=50)  # You can adjust n_components based on your use case
    svd_matrix = svd.fit_transform(user_item_matrix)
    
    # Create the similarity matrix
    similarity_matrix = pd.DataFrame(svd_matrix).dot(svd_matrix.T)
    
    # Create a list of item ids (product IDs)
    item_ids = user_item_matrix.columns
    return similarity_matrix, item_ids

def recommend_products_svd(user_id, similarity_matrix, item_ids, top_n=5):
    """
    Recommend products based on the SVD similarity matrix.
    """
    # Get the user's ratings (interactions) from the user-item matrix
    user_ratings = similarity_matrix.loc[user_id]

    # Sort by similarity scores, and recommend the top_n products
    recommended = user_ratings.sort_values(ascending=False).head(top_n)
    
    # Create a dictionary of recommended products and their similarity score
    recommended_products = {item_ids[i]: recommended[i] for i in range(len(recommended))}
    
    return recommended_products

def train_collaborative_filtering(request_user):
    """
    Trains a collaborative filtering model and returns recommended product IDs.
    """
    # Get the user-item matrix for collaborative filtering
    user_item_matrix = get_user_item_matrix()

    # Build the SVD model and similarity matrix
    similarity_matrix, item_ids = build_svd_model(user_item_matrix)

    # Get top recommendations for each user based on the similarity matrix
    recommendations = {}
    for user_id in user_item_matrix.index:
        recommendations[user_id] = recommend_products_svd(user_id, similarity_matrix, item_ids)

    # Flatten the recommendations into a list of unique product IDs for the requested user
    user_recommendations = recommendations.get(request_user.id, {})  # Get recommendations for the specific user
    recommended_product_ids = list(user_recommendations.keys())

    return user_recommendations, recommended_product_ids

def recommend_products_autoencoder(user_id, model, encoder, user_item_matrix, top_n=5):
    """
    Get top N product recommendations for a user using autoencoder-based collaborative filtering.
    
    Parameters:
    - user_id (int): The user ID for whom to generate recommendations.
    - model (keras.Model): The trained autoencoder model.
    - encoder (keras.Model): The encoder part of the autoencoder model.
    - user_item_matrix (pd.DataFrame): The user-item interaction matrix.
    - top_n (int): The number of top recommendations to return.
    
    Returns:
    - recommended_product_ids (list): A list of recommended product IDs.
    """
    
    # Get the user-item interaction row for the given user_id
    user_row = user_item_matrix.iloc[user_id - 1].values  # Assumes user_id is 1-indexed

    # Use the encoder to get the latent representation (embedding) of the user's interactions
    user_latent_vector = encoder.predict(np.expand_dims(user_row, axis=0))
    
    # Use the encoder to get latent representations (embeddings) for all products
    product_latent_vectors = encoder.predict(user_item_matrix.values)

    # Calculate similarity between the user's latent vector and all product latent vectors (cosine similarity)
    similarity_scores = cosine_similarity(user_latent_vector, product_latent_vectors)

    # Get the top N products based on similarity scores
    similar_product_indices = similarity_scores.argsort()[0][::-1][:top_n]
    recommended_product_ids = user_item_matrix.columns[similar_product_indices].tolist()

    return recommended_product_ids

def get_recommendations(user_id, user_item_matrix, top_n=5):
    """
    Get combined recommendations (SVD + Association Rules) for the user.
    """
    logger.info(f"Getting recommendations for user {user_id}...")

    # Get the collaborative filtering recommendations
    model, encoder = train_autoencoder(user_item_matrix, encoding_dim=50, epochs=10)
    cf_recommendations = recommend_products_autoencoder(user_id, model, encoder, user_item_matrix, top_n)
    logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")

    # Get the association rule-based recommendations
    rules = generate_association_rules(min_support=0.01, min_threshold=1.0)
    logger.info(f"Generated association rules: {rules.shape[0]} rules found.")
    ar_recommendations = get_recommendations_based_on_rules(rules, user_id, top_n)
    logger.info(f"Association rule recommendations: {ar_recommendations}")

    # Combine the recommendations
    recommended_products = {}

    # Add recommendations from association rules
    for product in ar_recommendations:
        recommended_products[product.id] = product

    # Add recommendations from collaborative filtering
    for product_id in cf_recommendations:
        try:
            product = Product.objects.get(id=product_id)
            recommended_products[product.id] = product
        except Product.DoesNotExist:
            logger.warning(f"Product with ID {product_id} not found.")
    
    logger.info(f"Total recommendations: {len(recommended_products)}")
    return list(recommended_products.values())

def recommend_products_svd(user_id, similarity_matrix, item_ids, top_n=5):
    """
    Get top_n recommendations using Collaborative Filtering (SVD).
    """
    recommendations = {}
    user_index = user_id - 1  # Assuming user_id is 1-indexed

    similar_items = similarity_matrix[user_index]
    similar_indices = similar_items.argsort()[::-1][1:top_n+1]

    for i in similar_indices:
        recommendations[item_ids[i]] = similar_items[i]

    return recommendations

def generate_association_rules(min_support=0.01, min_threshold=1.0):
    """
    Generates association rules using the Apriori algorithm for product recommendations.
    """
    transactions = Transaction.objects.all().values('user_id', 'product_id')
    df = pd.DataFrame(transactions)

    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)

    basket = df.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    return rules


def train_autoencoder(user_item_matrix, encoding_dim=50, epochs=10):
    """
    Train an autoencoder model to learn a latent representation for user-item interactions.
    
    Parameters:
    - user_item_matrix (pd.DataFrame): The user-item matrix (users x products)
    - encoding_dim (int): The dimension of the latent encoding
    - epochs (int): The number of training epochs
    
    Returns:
    - model (keras.Model): The trained autoencoder model
    - encoder (keras.Model): The encoder part of the autoencoder (for generating latent embeddings)
    """
    
    # Normalize the user-item matrix to the range [0, 1] (important for training neural networks)
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(user_item_matrix)
    
    # Define the input layer
    input_layer = Input(shape=(normalized_matrix.shape[1],))

    # Encoder
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Decoder
    decoded = Dense(normalized_matrix.shape[1], activation='sigmoid')(encoded)

    # Create the autoencoder model
    autoencoder = Model(input_layer, decoded)

    # Create the encoder model (just the part that encodes input data)
    encoder = Model(input_layer, encoded)

    # Compile the model
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

    # Train the autoencoder model
    autoencoder.fit(normalized_matrix, normalized_matrix, epochs=epochs, batch_size=256, shuffle=True, validation_split=0.1)
    
    return autoencoder, encoder


def get_recommendations_based_on_rules(rules, user_id, top_n=5):
    """
    Get product recommendations based on association rules.
    """
    transactions = Transaction.objects.filter(user_id=user_id)
    purchased_products = [transaction.product.id for transaction in transactions]

    recommended_products = set()

    for _, rule in rules.iterrows():
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        
        if any(item in purchased_products for item in antecedents):
            recommended_products.update(consequents)
        
        if len(recommended_products) >= top_n:
            break

    recommended_products = Product.objects.filter(id__in=list(recommended_products))[:top_n]
    return recommended_products

def train_collaborative_filtering():
    """
    Trains a collaborative filtering model and returns recommended product IDs.
    """
    # Get the user-item matrix for collaborative filtering
    user_item_matrix = get_user_item_matrix()

    # Build the SVD model and similarity matrix
    similarity_matrix, item_ids = build_svd_model(user_item_matrix)

    # Get top recommendations for each user based on the similarity matrix
    recommendations = {}
    for user_id in user_item_matrix.index:
        recommendations[user_id] = recommend_products_svd(user_id, similarity_matrix, item_ids)

    # Flatten the recommendations into a list of unique product IDs
    recommended_product_ids = list(set([product_id for user_recommendations in recommendations.values() for product_id in user_recommendations.keys()]))

    return recommendations, recommended_product_ids

def recommend_products_autoencoder(user_id, model, encoder, user_item_matrix, top_n=5):
    """
    Get top N product recommendations for a user using autoencoder-based collaborative filtering.
    
    Parameters:
    - user_id (int): The user ID for whom to generate recommendations.
    - model (keras.Model): The trained autoencoder model.
    - encoder (keras.Model): The encoder part of the autoencoder model.
    - user_item_matrix (pd.DataFrame): The user-item interaction matrix.
    - top_n (int): The number of top recommendations to return.
    
    Returns:
    - recommended_product_ids (list): A list of recommended product IDs.
    """
    
    # Get the user-item interaction row for the given user_id
    user_row = user_item_matrix.iloc[user_id - 1].values  # Assumes user_id is 1-indexed

    # Use the encoder to get the latent representation (embedding) of the user's interactions
    user_latent_vector = encoder.predict(np.expand_dims(user_row, axis=0))
    
    # Use the encoder to get latent representations (embeddings) for all products
    product_latent_vectors = encoder.predict(user_item_matrix.values)

    # Calculate similarity between the user's latent vector and all product latent vectors (cosine similarity)
    similarity_scores = cosine_similarity(user_latent_vector, product_latent_vectors)

    # Get the top N products based on similarity scores
    similar_product_indices = similarity_scores.argsort()[0][::-1][:top_n]
    recommended_product_ids = user_item_matrix.columns[similar_product_indices].tolist()

    return recommended_product_ids

def get_recommendations(user_id, user_item_matrix, top_n=5):
    """
    Get combined recommendations (SVD + Association Rules) for the user.
    """
    logger.info(f"Getting recommendations for user {user_id}...")

    # Get the collaborative filtering recommendations
    model, encoder = train_autoencoder(user_item_matrix, encoding_dim=50, epochs=10)
    cf_recommendations = recommend_products_autoencoder(user_id, model, encoder, user_item_matrix, top_n)
    logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")

    # Get the association rule-based recommendations
    rules = generate_association_rules(min_support=0.01, min_threshold=1.0)
    logger.info(f"Generated association rules: {rules.shape[0]} rules found.")
    ar_recommendations = get_recommendations_based_on_rules(rules, user_id, top_n)
    logger.info(f"Association rule recommendations: {ar_recommendations}")

    # Combine the recommendations
    recommended_products = {}

    # Add recommendations from association rules
    for product in ar_recommendations:
        recommended_products[product.id] = product

    # Add recommendations from collaborative filtering
    for product_id in cf_recommendations:
        try:
            product = Product.objects.get(id=product_id)
            recommended_products[product.id] = product
        except Product.DoesNotExist:
            logger.warning(f"Product with ID {product_id} not found.")
    
    logger.info(f"Total recommendations: {len(recommended_products)}")
    return list(recommended_products.values())
