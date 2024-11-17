import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from django.db.models import Count
from products.models import Cart, Transaction, Product
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

# Set up logging
logger = logging.getLogger(__name__)

def create_user_item_matrix():
    """
    Creates a user-item matrix using both cart and transaction data.
    The matrix values will be 1 if the user has interacted with the product (either in cart or transaction).
    """
    try:
        # Step 1: Get all transaction data
        transactions = Transaction.objects.all().values('user_id', 'product_id')
        transaction_df = pd.DataFrame(transactions)

        logger.info(f"Transactions data retrieved: {len(transaction_df)} records found.")
        
        if transaction_df.empty:
            logger.warning("No transaction data found.")
        
        if 'user_id' not in transaction_df.columns or 'product_id' not in transaction_df.columns:
            logger.error("Missing required columns ('user_id', 'product_id') in transaction data.")
            return pd.DataFrame(), None

        transaction_df['interaction'] = 1  # Purchases get an interaction score of 1

        # Step 2: Get all cart data
        cart_items = Cart.objects.all().values('user_id', 'product_id')
        cart_df = pd.DataFrame(cart_items)

        logger.info(f"Cart data retrieved: {len(cart_df)} records found.")
        
        if cart_df.empty:
            logger.warning("No cart data found.")
        
        if 'user_id' not in cart_df.columns or 'product_id' not in cart_df.columns:
            logger.error("Missing required columns ('user_id', 'product_id') in cart data.")
            return pd.DataFrame(), None

        cart_df['interaction'] = 0.5  # Cart items get a lower interaction score (0.5)

        # Step 3: Combine transaction and cart data
        if transaction_df.empty and cart_df.empty:
            logger.warning("Both transaction and cart data are empty. Cannot create user-item matrix.")
            return pd.DataFrame(), None  # Return empty if no data is available
        
        df = pd.concat([transaction_df[['user_id', 'product_id', 'interaction']], 
                        cart_df[['user_id', 'product_id', 'interaction']]], ignore_index=True)

        logger.info(f"Combined data (transactions + cart): {len(df)} records found.")

        # Step 4: Create user-item matrix
        user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='interaction', aggfunc='sum', fill_value=0)

        logger.info(f"User-item matrix created with shape: {user_item_matrix.shape}")

        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. No recommendations available.")
            return pd.DataFrame(), None  # Return empty DataFrame if matrix is empty

        # Visualize the user-item matrix as a heatmap
        user_item_matrix_plot = plot_user_item_matrix(user_item_matrix)

        return user_item_matrix, user_item_matrix_plot  # Return both data and plot

    except Exception as e:
        logger.error(f"Error occurred while creating user-item matrix: {e}")
        return pd.DataFrame(), None  # Return empty DataFrame and None if error occurs

def plot_user_item_matrix(user_item_matrix):
    """Visualizes the user-item matrix as a heatmap and returns a base64-encoded image URL."""
    try:
        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. Cannot generate heatmap.")
            return None  # Return None if the matrix is empty

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
        # Step 1: Get KNN-based recommendations
        knn_recommendations = get_knn_recommendations(user_id, top_n)
        
        if not knn_recommendations:
            logger.warning(f"No KNN-based recommendations found for user {user_id}. Falling back to association rules.")
            
            # Step 2: If KNN fails, use Association Rules to generate recommendations
            association_recommendations = get_association_rule_recommendations(user_id, top_n)
            
            if not association_recommendations:
                logger.warning(f"No recommendations found using association rules for user {user_id}. Falling back to popular products.")
                
                # Step 3: If association rules fail, fall back to popular products
                popular_recommendations = get_popular_products(top_n)
                plot_popular_products(top_n)  # Visualize popular products
                recommended_products = Product.objects.filter(id__in=popular_recommendations)
                return recommended_products
        
            # If association rule recommendations are found, use them
            return association_recommendations
        
        # If KNN recommendations are found, use them
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

def get_association_rule_recommendations(user_id, top_n=5):
    """
    Generate recommendations using association rule mining (Apriori algorithm).
    """
    try:
        # Step 1: Fetch transaction data (user_id, product_id)
        transactions = Transaction.objects.all().values('user_id', 'product_id')
        transaction_df = pd.DataFrame(transactions)

        if transaction_df.empty:
            logger.warning("No transaction data found.")
            return []

        # Step 2: Create a transaction matrix (one-hot encoding)
        transaction_matrix = transaction_df.pivot_table(index='user_id', columns='product_id', values='product_id', aggfunc='count', fill_value=0)
        
        # Step 3: Apply Apriori algorithm to find frequent itemsets
        frequent_itemsets = apriori(transaction_matrix, min_support=0.05, use_colnames=True)
        
        # Step 4: Generate association rules from the frequent itemsets
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

        # Step 5: Filter rules with high confidence and lift
        if rules.empty:
            logger.warning("No strong association rules found.")
            return []

        # Sort the rules based on confidence and lift
        rules = rules.sort_values(by=['confidence', 'lift'], ascending=False)

        # Step 6: Recommend products based on the rules
        recommendations = []
        for _, row in rules.iterrows():
            # Check if the user has already bought the product on the left-hand side of the rule
            lhs = list(row['antecedents'])  # Left-hand side (product(s) bought first)
            rhs = list(row['consequents'])  # Right-hand side (product(s) bought after)

            if rhs and lhs:
                # Find products the user has bought (for user_id)
                user_purchases = transaction_df[transaction_df['user_id'] == user_id]['product_id'].tolist()
                
                # If the user has purchased any product in the left-hand side of the rule, recommend the right-hand side
                if any(product in user_purchases for product in lhs):
                    recommendations.extend(rhs)

            if len(recommendations) >= top_n:
                break

        # Remove duplicates and return the top_n recommended products
        recommendations = list(set(recommendations))
        recommendations = recommendations[:top_n]
        
        logger.info(f"Association rule recommendations for user {user_id}: {recommendations}")
        return Product.objects.filter(id__in=recommendations)

    except Exception as e:
        logger.error(f"Error occurred during association rule mining: {e}")
        return []  # Return empty list if any error occurs
