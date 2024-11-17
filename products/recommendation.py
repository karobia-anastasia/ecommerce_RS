import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from products.models import Cart, Transaction, Product
from django.db.models import Count
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import matplotlib.pyplot as plt
# Set up logging
logger = logging.getLogger(__name__)
def create_user_item_matrix():
    """
    Creates a user-item matrix using both cart and transaction data.
    The matrix values will be 1 if the user has interacted with the product (either in cart or transaction).
    """
    try:
        # Get all transaction data
        transactions = Transaction.objects.all().values('user_id', 'product_id')
        transaction_df = pd.DataFrame(transactions)

        # Log the transaction data to check
        if transaction_df.empty:
            logger.warning("No transaction data found.")
        
        transaction_df['interaction'] = 1  # Purchases get an interaction score of 1

        # Get all cart data
        cart_items = Cart.objects.all().values('user_id', 'product_id')
        cart_df = pd.DataFrame(cart_items)

        # Log the cart data to check
        if cart_df.empty:
            logger.warning("No cart data found.")
        
        cart_df['interaction'] = 0.5  # Cart items get a lower interaction score (0.5)

        # Combine data from both transactions and cart
        if transaction_df.empty and cart_df.empty:
            logger.warning("Both transaction and cart data are empty. Cannot create user-item matrix.")
            return pd.DataFrame(), None  # Return empty if no data is available
        
        df = pd.concat([transaction_df[['user_id', 'product_id', 'interaction']], 
                        cart_df[['user_id', 'product_id', 'interaction']]], ignore_index=True)

        # Create user-item matrix
        user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='interaction', aggfunc='sum', fill_value=0)

        # Log the shape of the user-item matrix
        logger.info(f"User-item matrix created with shape: {user_item_matrix.shape}")

        # Check if the user-item matrix is empty
        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. Cannot generate heatmap.")
            return user_item_matrix, None  # Return empty matrix and None if empty
        
        # Generate the heatmap plot
        user_item_matrix_plot = plot_user_item_matrix(user_item_matrix)
        
        return user_item_matrix, user_item_matrix_plot

    except Exception as e:
        logger.error(f"Error occurred while creating user-item matrix: {e}")
        return pd.DataFrame(), None  # Return empty DataFrame and None if error occurs

def plot_user_item_matrix(user_item_matrix):
    """ Visualizes the user-item matrix as a heatmap and returns a base64-encoded image URL. """
    try:
        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. Cannot generate heatmap.")
            return None  # Return None if the matrix is empty

        # Generate the heatmap plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(user_item_matrix, cmap="YlGnBu", annot=False, cbar=True)
        plt.title("User-Item Interaction Matrix Heatmap")
        plt.xlabel("Product ID")
        plt.ylabel("User ID")

        # Save the plot to a BytesIO buffer and encode it as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url  # Return the base64 encoded string for rendering in the template

    except Exception as e:
        logger.error(f"Error occurred while plotting user-item matrix: {e}")
        return None

def get_knn_recommendations(user_id, top_n=5):
    """Generates KNN-based recommendations for a given user."""
    try:
        user_item_matrix, _ = create_user_item_matrix()
        
        # If the matrix is empty, return an empty list
        if user_item_matrix.empty:
            logger.warning(f"User-item matrix is empty for user {user_id}. Cannot generate KNN recommendations.")
            return []

        # Generate KNN-based recommendations
        recommended_product_ids = knn_recommend_products(user_id, user_item_matrix, top_n)
        
        if not recommended_product_ids:
            logger.warning(f"No recommendations generated for user {user_id} using KNN.")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating KNN recommendations for user {user_id}: {e}")
        return []  # Return an empty list if any error occurs

def knn_recommend_products(user_id, user_item_matrix, top_n=5):
    """Generate KNN-based product recommendations for a given user."""
    try:
        if user_id not in user_item_matrix.index:
            logger.warning(f"User {user_id} not found in user-item matrix. Cannot generate KNN recommendations.")
            return []

        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')
        knn.fit(user_item_matrix.values.T)  # Fit on products (columns)

        distances, indices = knn.kneighbors(user_vector)

        # Exclude the user from the recommendations
        recommended_product_ids = user_item_matrix.columns[indices.flatten()[1:]].tolist()

        logger.info(f"KNN recommendations for user {user_id}: {recommended_product_ids}")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating KNN recommendations: {e}")
        return []  # Return an empty list if any error occurs

def get_combined_recommendations(user_id, top_n=5):
    """
    Combines recommendations from KNN and fallback to popular products if necessary.
    """
    try:
        # Get KNN-based recommendations
        knn_recommendations = get_knn_recommendations(user_id, top_n)

        if not knn_recommendations:
            logger.warning(f"No KNN-based recommendations found for user {user_id}. Falling back to popular products.")
            popular_recommendations = get_popular_products(top_n)
            plot_popular_products(top_n)  # Visualize popular products
            recommended_products = Product.objects.filter(id__in=popular_recommendations)
            return recommended_products
        
        # Fetch Product objects for KNN-based recommendations
        recommended_products = Product.objects.filter(id__in=knn_recommendations)
        return recommended_products

    except Exception as e:
        logger.error(f"Error occurred while fetching combined recommendations for user {user_id}: {e}")
        return []  # Return an empty list if any error occurs

def get_popular_products(top_n=5):
    """Returns the top N most purchased products."""
    try:
        popular_products = Product.objects.annotate(purchase_count=Count('transaction')).order_by('-purchase_count')[:top_n]
        return [product.id for product in popular_products]
    
    except Exception as e:
        logger.error(f"Error occurred while fetching popular products: {e}")
        return []  # Return an empty list if any error occurs

def plot_popular_products(top_n=5):
    """Visualizes the top N popular products based on purchase count."""
    try:
        popular_products = Product.objects.annotate(purchase_count=Count('transaction')).order_by('-purchase_count')[:top_n]
        popular_product_names = [product.name for product in popular_products]
        purchase_counts = [product.purchase_count for product in popular_products]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=popular_product_names, y=purchase_counts)
        plt.title(f"Top {top_n} Popular Products")
        plt.xlabel("Product Name")
        plt.ylabel("Number of Purchases")
        plt.xticks(rotation=45, ha="right")

        # Save plot to a BytesIO object and encode as base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close()

        return img_base64

    except Exception as e:
        logger.error(f"Error occurred while plotting popular products: {e}")
        return None
