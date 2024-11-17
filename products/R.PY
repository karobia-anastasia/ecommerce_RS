


import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
from mlxtend.frequent_patterns import apriori, association_rules
from products.models import Cart, Transaction, Product
import logging

logger = logging.getLogger(__name__)

# ------------ Preprocessing Functions ------------

def get_user_item_matrix():
    """
    Create a user-item matrix based on add-to-cart and purchase interactions.
    We use different weights for each action:
    - Add to Cart: 1
    - Purchase: 5
    """
    # Fetch data from Cart and Transaction models
    cart_data = Cart.objects.all().values('user__id', 'product', 'quantity')
    purchase_data = Transaction.objects.all().values('user__id', 'product', 'quantity')

    cart_df = pd.DataFrame(cart_data)
    cart_df['score'] = 1  # Add to Cart gets 1-point feedback

    purchase_df = pd.DataFrame(purchase_data)
    purchase_df['score'] = 5  # Purchased products get 5-point feedback

    combined_df = pd.concat([cart_df[['user__id', 'product', 'score']],
                             purchase_df[['user__id', 'product', 'score']]])

    # Create the user-item matrix (pivot table)
    user_item_matrix = combined_df.pivot_table(index='user__id', columns='product', values='score', aggfunc='sum', fill_value=0)

    return user_item_matrix

def preprocess_data():
    """
    Preprocess the data and return the user-item matrix for training.
    """
    return get_user_item_matrix()


# ------------ Collaborative Filtering Model (SVD) ------------

def build_svd_model():
    """
    Build a recommendation model using collaborative filtering (SVD).
    """
    user_item_matrix = preprocess_data()

    # Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix = svd.fit_transform(user_item_matrix)

    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(matrix)

    return similarity_matrix, user_item_matrix.columns

def get_top_n_recommendations(similarity_matrix, item_ids, top_n=5):
    """
    For each item (product), recommend top_n similar products.
    """
    recommendations = {}

    for i, item_id in enumerate(item_ids):
        similar_items = similarity_matrix[i]
        similar_indices = similar_items.argsort()[::-1][1:top_n+1]  # Exclude self (highest similarity)
        recommendations[item_id] = [item_ids[j] for j in similar_indices]

    return recommendations


# ------------ Association Rules ------------

def generate_association_rules(min_support=0.01, min_threshold=1.0):
    """
    Generates association rules using the Apriori algorithm for older versions of `mlxtend`.
    """
    # Fetch transactions from the database
    transactions = Transaction.objects.all().values('user_id', 'product_id')

    df = pd.DataFrame(transactions)

    # Convert user_id and product_id to strings
    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)

    # Create a user-product basket matrix
    basket = df.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    
    # Convert to binary matrix where 1 represents the product being bought by the user
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Generate frequent itemsets using apriori
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    # Generate association rules using the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    return rules


def get_recommendations_based_on_rules(rules, user_id, top_n=5):
    """
    Get product recommendations based on association rules.
    """
    # Fetch the user's purchase history from the database
    transactions = Transaction.objects.filter(user_id=user_id)

    # Get the list of products the user has purchased
    purchased_products = [transaction.product.id for transaction in transactions]

    # Initialize an empty set to hold recommendations
    recommended_products = set()

    # Iterate through the rules to find products frequently bought together with the user's purchased items
    for _, rule in rules.iterrows():
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        
        # If the user has purchased any product from the antecedents, recommend products from the consequents
        if any(item in purchased_products for item in antecedents):
            recommended_products.update(consequents)
        
        # Stop when we have enough recommendations
        if len(recommended_products) >= top_n:
            break
    
    # Convert the recommended product IDs to Product objects (this is assuming you have a Product model)
    recommended_products = Product.objects.filter(id__in=list(recommended_products))[:top_n]

    return recommended_products


# ------------ Recommendation Logic ------------

def get_recommendations(user_id, model, product_ids, top_n=5):
    logger.info(f"Getting recommendations for user {user_id}...")

    # Get Collaborative Filtering Recommendations
    cf_recommendations = recommend_products(user_id, model, product_ids)
    logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")

    # Get Association Rule-based Recommendations
    rules = generate_association_rules()
    logger.info(f"Generated association rules: {rules.shape[0]} rules found.")

    ar_recommendations = get_recommendations_based_on_rules(rules, user_id)
    logger.info(f"Association rule recommendations: {ar_recommendations}")

    # Combine recommendations
    recommended_products = {}

    # First, add Association Rule recommendations (which are Product objects)
    for product in ar_recommendations:
        recommended_products[product.id] = product

    # Then, add Collaborative Filtering recommendations (which are just IDs)
    for product_id in cf_recommendations:
        try:
            product = Product.objects.get(id=product_id)  # Fetch the actual Product object
            recommended_products[product.id] = product  # Add to the recommendations
        except Product.DoesNotExist:
            logger.warning(f"Product with ID {product_id} not found.")
    
    logger.info(f"Total recommendations: {len(recommended_products)}")
    return list(recommended_products.values())


# ------------ Collaborative Filtering Model Training ------------
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
from mlxtend.frequent_patterns import apriori, association_rules
from products.models import Cart, Transaction, Product
import logging

logger = logging.getLogger(__name__)

# ------------ Preprocessing Functions ------------

def get_user_item_matrix():
    """
    Create a user-item matrix based on add-to-cart and purchase interactions.
    We use different weights for each action:
    - Add to Cart: 1
    - Purchase: 5
    """
    # Fetch data from Cart and Transaction models
    cart_data = Cart.objects.all().values('user__id', 'product', 'quantity')
    purchase_data = Transaction.objects.all().values('user__id', 'product', 'quantity')

    cart_df = pd.DataFrame(cart_data)
    cart_df['score'] = 1  # Add to Cart gets 1-point feedback

    purchase_df = pd.DataFrame(purchase_data)
    purchase_df['score'] = 5  # Purchased products get 5-point feedback

    combined_df = pd.concat([cart_df[['user__id', 'product', 'score']],
                             purchase_df[['user__id', 'product', 'score']]])

    # Create the user-item matrix (pivot table)
    user_item_matrix = combined_df.pivot_table(index='user__id', columns='product', values='score', aggfunc='sum', fill_value=0)

    return user_item_matrix

def preprocess_data():
    """
    Preprocess the data and return the user-item matrix for training.
    """
    return get_user_item_matrix()


# ------------ Collaborative Filtering Model (SVD) ------------

def build_svd_model():
    """
    Build a recommendation model using collaborative filtering (SVD).
    """
    user_item_matrix = preprocess_data()

    # Apply Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix = svd.fit_transform(user_item_matrix)

    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(matrix)

    return similarity_matrix, user_item_matrix.columns

def get_top_n_recommendations(similarity_matrix, item_ids, top_n=5):
    """
    For each item (product), recommend top_n similar products.
    """
    recommendations = {}

    for i, item_id in enumerate(item_ids):
        similar_items = similarity_matrix[i]
        similar_indices = similar_items.argsort()[::-1][1:top_n+1]  # Exclude self (highest similarity)
        recommendations[item_id] = [item_ids[j] for j in similar_indices]

    return recommendations


# ------------ Association Rules ------------

def generate_association_rules(min_support=0.01, min_threshold=1.0):
    """
    Generates association rules using the Apriori algorithm for older versions of `mlxtend`.
    """
    # Fetch transactions from the database
    transactions = Transaction.objects.all().values('user_id', 'product_id')

    df = pd.DataFrame(transactions)

    # Convert user_id and product_id to strings
    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)

    # Create a user-product basket matrix
    basket = df.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    
    # Convert to binary matrix where 1 represents the product being bought by the user
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Generate frequent itemsets using apriori
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    # Generate association rules using the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    return rules


def get_recommendations_based_on_rules(rules, user_id, top_n=5):
    """
    Get product recommendations based on association rules.
    """
    # Fetch the user's purchase history from the database
    transactions = Transaction.objects.filter(user_id=user_id)

    # Get the list of products the user has purchased
    purchased_products = [transaction.product.id for transaction in transactions]

    # Initialize an empty set to hold recommendations
    recommended_products = set()

    # Iterate through the rules to find products frequently bought together with the user's purchased items
    for _, rule in rules.iterrows():
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        
        # If the user has purchased any product from the antecedents, recommend products from the consequents
        if any(item in purchased_products for item in antecedents):
            recommended_products.update(consequents)
        
        # Stop when we have enough recommendations
        if len(recommended_products) >= top_n:
            break
    
    # Convert the recommended product IDs to Product objects (this is assuming you have a Product model)
    recommended_products = Product.objects.filter(id__in=list(recommended_products))[:top_n]

    return recommended_products


# ------------ Recommendation Logic ------------

def get_recommendations(user_id, model, product_ids, top_n=5):
    logger.info(f"Getting recommendations for user {user_id}...")

    # Get Collaborative Filtering Recommendations
    cf_recommendations = recommend_products(user_id, model, product_ids)
    logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")

    # Get Association Rule-based Recommendations
    rules = generate_association_rules()
    logger.info(f"Generated association rules: {rules.shape[0]} rules found.")

    ar_recommendations = get_recommendations_based_on_rules(rules, user_id)
    logger.info(f"Association rule recommendations: {ar_recommendations}")

    # Combine recommendations
    recommended_products = {}

    # First, add Association Rule recommendations (which are Product objects)
    for product in ar_recommendations:
        recommended_products[product.id] = product

    # Then, add Collaborative Filtering recommendations (which are just IDs)
    for product_id in cf_recommendations:
        try:
            product = Product.objects.get(id=product_id)  # Fetch the actual Product object
            recommended_products[product.id] = product  # Add to the recommendations
        except Product.DoesNotExist:
            logger.warning(f"Product with ID {product_id} not found.")
    
    logger.info(f"Total recommendations: {len(recommended_products)}")
    return list(recommended_products.values())


# ------------ Collaborative Filtering Model Training ------------
import pandas as pd
from surprise import SVD, Reader, Dataset
import logging

logger = logging.getLogger(__name__)

def train_collaborative_filtering():
    """
    Train a collaborative filtering model using matrix factorization (SVD).
    This function loads transaction data from the database, prepares it for training,
    and returns the trained model and a list of unique product IDs.
    """
    # Fetch all transactions from the database
    transactions = Transaction.objects.all()
    logger.info(f"Fetched {len(transactions)} transactions from the database.")

    # Prepare the data for training
    data = {
        'user_id': [],
        'product_id': [],
        'rating': []  # This will be filled with the total_amount instead of ratings
    }

    # Collect data
    for trans in transactions:
        if trans.user and trans.product and isinstance(trans.total_amount, (int, float)):
            user_id = trans.user.id
            product_id = trans.product.id
            rating = trans.total_amount  # Use total_amount as the rating

            data['user_id'].append(user_id)
            data['product_id'].append(product_id)
            data['rating'].append(rating)
        else:
            logger.warning(f"Skipping transaction due to invalid or missing data: {trans.id}")

    # Check if the required data is available
    if not data['user_id']:
        logger.error("No valid transaction data available for collaborative filtering.")
        raise ValueError("No valid transaction data found.")

    # Convert to a DataFrame for easier manipulation
    df = pd.DataFrame(data)

    logger.info(f"DataFrame head:\n{df.head()}")

    # Determine the rating scale (from the minimum to the maximum total_amount)
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()

    # Prepare the dataset for the Surprise library
    reader = Reader(rating_scale=(min_rating, max_rating))  # Define the rating scale based on total_amount
    dataset = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

    # Build the training set (the entire dataset in this case)
    trainset = dataset.build_full_trainset()

    # Initialize the Singular Value Decomposition (SVD) model for collaborative filtering
    model = SVD()

    # Train the model
    model.fit(trainset)

    # Extract the unique product IDs for future recommendations
    product_ids = df['product_id'].unique()

    logger.info(f"Found {len(product_ids)} unique product IDs.")

    return model, product_ids

def recommend_products(user_id, model, product_ids, top_n=5):
    """
    Get recommendations using collaborative filtering (SVD).
    """
    cf_recommendations = []

    for product_id in product_ids:
        if model.trainset.knows_user(user_id) and model.trainset.knows_item(product_id):
            prediction = model.predict(user_id, product_id)  # Predict rating
            cf_recommendations.append((product_id, prediction.est))  # Store product_id and predicted rating
        else:
            logger.warning(f"User {user_id} or product {product_id} is not in the training data.")

    cf_recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating (highest first)

    return [product[0] for product in cf_recommendations[:top_n]]  # Return top_n products


# ------------ Main Execution ------------

if __name__ == "__main__":
    # Train collaborative filtering model
    cf_model, product_ids = train_collaborative_filtering()

    # Example: Get recommendations for user 123
    user_id = 123
    recommendations = get_recommendations(user_id, cf_model, product_ids, top_n=5)

    # Print or return the recommended products
    logger.info(f"Recommended products for user {user_id}: {recommendations}")

def recommend_products(user_id, model, product_ids, top_n=5):
    """
    Get recommendations using collaborative filtering (SVD).
    """
    cf_recommendations = []

    for product_id in product_ids:
        if model.trainset.knows_user(user_id) and model.trainset.knows_item(product_id):
            prediction = model.predict(user_id, product_id)  # Predict rating
            cf_recommendations.append((product_id, prediction.est))  # Store product_id and predicted rating
        else:
            logger.warning(f"User {user_id} or product {product_id} is not in the training data.")

    cf_recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating (highest first)

    return [product[0] for product in cf_recommendations[:top_n]]  # Return top_n products


# ------------ Main Execution ------------

if __name__ == "__main__":
    # Train collaborative filtering model
    cf_model, product_ids = train_collaborative_filtering()

    # Example: Get recommendations for user 123
    user_id = 123
    recommendations = get_recommendations(user_id, cf_model, product_ids, top_n=5)

    # Print or return the recommended products
    logger.info(f"Recommended products for user {user_id}: {recommendations}")






# import pandas as pd
# import numpy as np
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from surprise import SVD, Reader, Dataset
# from mlxtend.frequent_patterns import apriori, association_rules
# from products.models import Cart, Transaction, Product
# import logging

# logger = logging.getLogger(__name__)

# # ------------ Preprocessing Functions ------------

# def get_user_item_matrix():
#     """
#     Create a user-item matrix based on add-to-cart and purchase interactions.
#     We use different weights for each action:
#     - Add to Cart: 1
#     - Purchase: 5
#     """
#     # Fetch data from Cart and Transaction models
#     cart_data = Cart.objects.all().values('user__id', 'product', 'quantity')
#     purchase_data = Transaction.objects.all().values('user__id', 'product', 'quantity')

#     cart_df = pd.DataFrame(cart_data)
#     cart_df['score'] = 1  # Add to Cart gets 1-point feedback

#     purchase_df = pd.DataFrame(purchase_data)
#     purchase_df['score'] = 5  # Purchased products get 5-point feedback

#     combined_df = pd.concat([cart_df[['user__id', 'product', 'score']],
#                              purchase_df[['user__id', 'product', 'score']]])

#     # Create the user-item matrix (pivot table)
#     user_item_matrix = combined_df.pivot_table(index='user__id', columns='product', values='score', aggfunc='sum', fill_value=0)

#     return user_item_matrix

# def preprocess_data():
#     """
#     Preprocess the data and return the user-item matrix for training.
#     """
#     return get_user_item_matrix()


# # ------------ Collaborative Filtering Model (SVD) ------------

# def build_svd_model():
#     """
#     Build a recommendation model using collaborative filtering (SVD).
#     """
#     user_item_matrix = preprocess_data()

#     # Apply Singular Value Decomposition (SVD)
#     svd = TruncatedSVD(n_components=50, random_state=42)
#     matrix = svd.fit_transform(user_item_matrix)

#     # Calculate the similarity matrix
#     similarity_matrix = cosine_similarity(matrix)

#     return similarity_matrix, user_item_matrix.columns

# def get_top_n_recommendations(similarity_matrix, item_ids, top_n=5):
#     """
#     For each item (product), recommend top_n similar products.
#     """
#     recommendations = {}

#     for i, item_id in enumerate(item_ids):
#         similar_items = similarity_matrix[i]
#         similar_indices = similar_items.argsort()[::-1][1:top_n+1]  # Exclude self (highest similarity)
#         recommendations[item_id] = [item_ids[j] for j in similar_indices]

#     return recommendations


# # ------------ Association Rules ------------

# def generate_association_rules(min_support=0.01, min_threshold=1.0):
#     """
#     Generates association rules using the Apriori algorithm for older versions of `mlxtend`.
#     """
#     # Fetch transactions from the database
#     transactions = Transaction.objects.all().values('user_id', 'product_id')

#     df = pd.DataFrame(transactions)

#     # Convert user_id and product_id to strings
#     df['user_id'] = df['user_id'].astype(str)
#     df['product_id'] = df['product_id'].astype(str)

#     # Create a user-product basket matrix
#     basket = df.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    
#     # Convert to binary matrix where 1 represents the product being bought by the user
#     basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

#     # Generate frequent itemsets using apriori
#     frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

#     # Generate association rules using the frequent itemsets
#     rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
#     return rules


# def get_recommendations_based_on_rules(rules, user_id, top_n=5):
#     """
#     Get product recommendations based on association rules.
#     """
#     # Fetch the user's purchase history from the database
#     transactions = Transaction.objects.filter(user_id=user_id)

#     # Get the list of products the user has purchased
#     purchased_products = [transaction.product.id for transaction in transactions]

#     # Initialize an empty set to hold recommendations
#     recommended_products = set()

#     # Iterate through the rules to find products frequently bought together with the user's purchased items
#     for _, rule in rules.iterrows():
#         antecedents = rule['antecedents']
#         consequents = rule['consequents']
        
#         # If the user has purchased any product from the antecedents, recommend products from the consequents
#         if any(item in purchased_products for item in antecedents):
#             recommended_products.update(consequents)
        
#         # Stop when we have enough recommendations
#         if len(recommended_products) >= top_n:
#             break
    
#     # Convert the recommended product IDs to Product objects (this is assuming you have a Product model)
#     recommended_products = Product.objects.filter(id__in=list(recommended_products))[:top_n]

#     return recommended_products


# # ------------ Recommendation Logic ------------

# def get_recommendations(user_id, model, product_ids, top_n=5):
#     logger.info(f"Getting recommendations for user {user_id}...")

#     # Get Collaborative Filtering Recommendations
#     cf_recommendations = recommend_products(user_id, model, product_ids)
#     logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")

#     # Get Association Rule-based Recommendations
#     rules = generate_association_rules()
#     logger.info(f"Generated association rules: {rules.shape[0]} rules found.")

#     ar_recommendations = get_recommendations_based_on_rules(rules, user_id)
#     logger.info(f"Association rule recommendations: {ar_recommendations}")

#     # Combine recommendations
#     recommended_products = {}

#     # First, add Association Rule recommendations (which are Product objects)
#     for product in ar_recommendations:
#         recommended_products[product.id] = product

#     # Then, add Collaborative Filtering recommendations (which are just IDs)
#     for product_id in cf_recommendations:
#         try:
#             product = Product.objects.get(id=product_id)  # Fetch the actual Product object
#             recommended_products[product.id] = product  # Add to the recommendations
#         except Product.DoesNotExist:
#             logger.warning(f"Product with ID {product_id} not found.")
    
#     logger.info(f"Total recommendations: {len(recommended_products)}")
#     return list(recommended_products.values())


# # ------------ Collaborative Filtering Model Training ------------

# def train_collaborative_filtering():
#     """
#     Train a collaborative filtering model using matrix factorization (SVD).
#     This function loads transaction data from the database, prepares it for training,
#     and returns the trained model and a list of unique product IDs.
#     """
#     # Fetch all transactions from the database
#     transactions = Transaction.objects.all()

#     logger.info(f"Fetched {len(transactions)} transactions from the database.")

#     # Prepare the data for training
#     data = {
#         'user_id': [],
#         'product_id': [],
#         'rating': []
#     }

#     # Collect data
#     for trans in transactions:
#         if trans.user and trans.product and isinstance(trans.total_amount, (int, float)):
#             user_id = trans.user.id
#             product_id = trans.product.id
#             rating = trans.total_amount  # Ensure it's numeric

#             data['user_id'].append(user_id)
#             data['product_id'].append(product_id)
#             data['rating'].append(rating)
#         else:
#             logger.warning(f"Skipping transaction due to invalid or missing data: {trans.id}")

#     # Convert to a DataFrame for easier manipulation
#     df = pd.DataFrame(data)

#     logger.info(f"DataFrame head:\n{df.head()}")

#     # Prepare the dataset for the Surprise library
#     reader = Reader(rating_scale=(0, df['rating'].max()))
#     dataset = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

#     # Build the training set (the entire dataset in this case)
#     trainset = dataset.build_full_trainset()

#     # Initialize the Singular Value Decomposition (SVD) model for collaborative filtering
#     model = SVD()

#     # Train the model
#     model.fit(trainset)

#     # Extract the unique product IDs for future recommendations
#     product_ids = df['product_id'].unique()

#     logger.info(f"Found {len(product_ids)} unique product IDs.")

#     return model, product_ids

# def recommend_products(user_id, model, product_ids, top_n=5):
#     """
#     Get recommendations using collaborative filtering (SVD).
#     """
#     cf_recommendations = []

#     for product_id in product_ids:
#         if model.trainset.knows_user(user_id) and model.trainset.knows_item(product_id):
#             prediction = model.predict(user_id, product_id)  # Predict rating
#             cf_recommendations.append((product_id, prediction.est))  # Store product_id and predicted rating
#         else:
#             logger.warning(f"User {user_id} or product {product_id} is not in the training data.")

#     cf_recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating (highest first)

#     return [product[0] for product in cf_recommendations[:top_n]]  # Return top_n products


# # ------------ Main Execution ------------

# if __name__ == "__main__":
#     # Train collaborative filtering model
#     cf_model, product_ids = train_collaborative_filtering()

#     # Example: Get recommendations for user 123
#     user_id = 123
#     recommendations = get_recommendations(user_id, cf_model, product_ids, top_n=5)

#     # Print or return the recommended products
#     logger.info(f"Recommended products for user {user_id}: {recommendations}")
