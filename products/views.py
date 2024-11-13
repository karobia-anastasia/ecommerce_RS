from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from products.form import ProductForm
from products.recommendation import *
from products.scrape import scrape_product
from .models import Order, Product, Transaction, Cart
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import numpy as np
import logging
from django.shortcuts import render, redirect
from .models import Product

logger = logging.getLogger(__name__)


@login_required
def add_to_cart_view(request, product_id):
    """
    Handle adding a product to the cart. If the product is already in the cart, update the quantity.
    """
    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))  # Default quantity is 1 if not specified

    # Get or create the cart item for the user and product
    cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)

    if not created:
        cart_item.quantity += quantity  # If the item is already in the cart, increase the quantity
        cart_item.save()

    return redirect('products:cart_view')  



@login_required
def update_cart_item(request, cart_item_id):
    """
    Handle updating the quantity of an item in the cart.
    """
    cart_item = get_object_or_404(Cart, id=cart_item_id, user=request.user)
    
    if request.method == 'POST':
        # Get the new quantity from the form
        new_quantity = int(request.POST.get('quantity', 1))

        # Ensure quantity is at least 1
        if new_quantity >= 1:
            cart_item.quantity = new_quantity
            cart_item.save()
    
    return redirect('products:cart_view')  

@login_required
def checkout(request):
    """
    Process the user's cart and create an order.

    Parameters:
        request: The HTTP request.

    Returns:
        redirect: Redirect to the order summary page after checkout.
    """
    cart_items = Cart.objects.filter(user=request.user)
    if not cart_items:
        return redirect('products:cart_view')  # Redirect to cart if there are no items in the cart

    total_price = sum(item.total_price() for item in cart_items)

    # Create an order for the user
    order = Order.objects.create(user=request.user, total_price=total_price)

    # Create transactions for each cart item
    for item in cart_items:
        transaction = Transaction.objects.create(
            user=request.user,
            product=item.product,
            quantity=item.quantity,
            price=item.product.price,
            total_amount=item.total_price(),
        )
        order.transactions.add(transaction)  # Corrected this line to use `transactions.add`

    # Clear the cart after the order is created
    cart_items.delete()

    # Redirect to a page where the user can view their order (you can customize this)
    return redirect('products:order_summary', order_id=order.id)



@login_required
def cart_view(request):
    """
    Displays the user's cart items and the total price.
    """
    cart_items = Cart.objects.filter(user=request.user)
    total = 0  # Initialize total to 0
    
    # Calculate the total price for the cart
    for item in cart_items:
        total += item.product.price * item.quantity
    
    return render(request, 'cart.html', {'cart_items': cart_items, 'total': total})

@login_required
def remove_from_cart(request, cart_item_id):
    """
    Remove an item from the cart.
    """
    cart_item = get_object_or_404(Cart, id=cart_item_id, user=request.user)
    cart_item.delete()
    return redirect('products:cart_view')  # Redirect back to the cart view after removal


@login_required
def order_summary(request, order_id):
    """
    Display the order summary after the user completes checkout.

    Parameters:
        request: The HTTP request.
        order_id (int): The ID of the order to display.

    Returns:
        render: The order summary page.
    """
    order = get_object_or_404(Order, id=order_id)
    return render(request, 'order_summary.html', {'order': order})

from django.shortcuts import render, redirect
from products.recommendation import train_collaborative_filtering, get_recommendations
from .models import Product
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
from .recommendation import train_collaborative_filtering, get_recommendations
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
from .recommendation import train_collaborative_filtering, get_recommendations
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
from .recommendation import train_collaborative_filtering
import logging

logger = logging.getLogger(__name__)

import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product

# Set up logging
logger = logging.getLogger(__name__)
import logging

# Set up logging
logger = logging.getLogger(__name__)

@login_required
def product_list(request):
    """
    Displays a list of products along with recommended products based on Collaborative Filtering.
    Scrapes new products from an external site if not already scraped, and loads recommendations.
    """
    if not request.user.is_authenticated:
        return redirect('login')  # Redirect to login page if user is not authenticated
    
    # List of external product URLs to scrape from
    external_product_urls = [
        'https://pricespy.co.uk/c/ps5-games',  # Example product URLs
        'https://pricespy.co.uk/fashion-accessories--c1944',
        'https://pricespy.co.uk/c/watches',
    ]

    # Scrape and save products from external URLs
    try:
        for url in external_product_urls:
            scrape_product(url)  # Scrape product details and save them to the database

        # After scraping, fetch the updated list of products
        products = Product.objects.all()

    except Exception as e:
        logger.error(f"Error scraping products: {e}")
        products = Product.objects.all()  # Continue with existing products if scraping fails

    # Load recommendations
    try:
        # Train the collaborative filtering model and get recommended product IDs
        model, product_ids = train_collaborative_filtering()

        # Fetch the recommended products from the database
        recommended_products = Product.objects.filter(id__in=product_ids)

        # Pass both products and recommended products to the template context
        context = {
            'products': products,
            'recommended_products': recommended_products
        }

    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")
        context = {
            'products': products,
            'recommended_products': []
        }

    # Render the page with both products and recommendations
    return render(request, 'product_list.html', context)

@login_required
def product_detail(request, pk):
    """
    Display the product details and show recommended products.

    Parameters:
        request: The HTTP request.
        pk (int): The product ID.

    Returns:
        render: The product detail page with recommended products.
    """
    product = get_object_or_404(Product, pk=pk)

    if request.method == 'POST' and 'add_to_cart' in request.POST:
        quantity = int(request.POST.get('quantity', 1))
        cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)

        if not created:
            cart_item.quantity += quantity
            cart_item.save()

        return redirect('product_detail', pk=product.pk)

    # Get recommended products (excluding current one)
    recommended_products = Product.objects.exclude(id=product.id)[:5]

    return render(request, 'product_detail.html', {
        'product': product,
        'recommended_products': recommended_products,
    })




# Product Create (Create new product)
def product_create(request):
    """
    Handle the creation of a new product.

    Parameters:
        request: The HTTP request.

    Returns:
        render: The product creation form page.
    """
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')
    else:
        form = ProductForm()
    return render(request, 'product_form.html', {'form': form})

# Product Edit (Edit existing product)
def product_edit(request, pk):
    """
    Handle editing an existing product.

    Parameters:
        request: The HTTP request.
        pk (int): The ID of the product to edit.

    Returns:
        render: The product editing form page.
    """
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')
    else:
        form = ProductForm(instance=product)
    return render(request, 'product_form.html', {'form': form})

# Product Delete (Delete a product)
def product_delete(request, pk):
    """
    Handle the deletion of a product.

    Parameters:
        request: The HTTP request.
        pk (int): The ID of the product to delete.

    Returns:
        redirect: Redirect to the product list after deletion.
    """
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        product.delete()
        return redirect('products:product_list')
    return render(request, 'delete_product.html', {'product': product})



# Set up logging
logger = logging.getLogger(__name__)

def visualize_recommendations(request):
    """
    Visualize product recommendations using both Collaborative Filtering (CF)
    and Association Rules (AR).

    Parameters:
        request: The HTTP request.

    Returns:
        render: The recommendations visualization page.
    """
    # Ensure user is authenticated
    user_id = request.user.id if request.user.is_authenticated else None
    if not user_id:
        return redirect('login')

    # Get Collaborative Filtering and Association Rule-based recommendations
    try:
        model, product_ids = train_collaborative_filtering()
        cf_recommendations = recommend_products(user_id, model, product_ids)
        logger.info(f"Collaborative filtering recommendations: {cf_recommendations}")
    except Exception as e:
        logger.error(f"Error getting collaborative filtering recommendations: {e}")
        cf_recommendations = []

    try:
        rules = generate_association_rules()
        ar_recommendations = get_recommendations_based_on_rules(rules, user_id)
        logger.info(f"Association rule recommendations: {ar_recommendations}")
    except Exception as e:
        logger.error(f"Error getting association rule recommendations: {e}")
        ar_recommendations = []

    # Combine both recommendations
    recommended_products = {product.id: product for product in ar_recommendations}
    for product in cf_recommendations:
        recommended_products[product.id] = product

    # Prepare data for visualization
    recommendations_list = list(recommended_products.values())
    if not recommendations_list:
        logger.warning("No recommendations to display.")

    # Generate Pie Chart for product categories
    category_counts = {}
    for product in recommendations_list:
        category = getattr(product.category, 'name', 'Uncategorized')  # Fallback if category is missing
        category_counts[category] = category_counts.get(category, 0) + 1

    if not category_counts:
        category_counts = {'No Category': 1}  # Fallback if no categories

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    pie_chart_img = base64.b64encode(img_buf.getvalue()).decode('utf-8')

    # Generate Bar Graph for top recommended products
    top_products = recommendations_list[:5]
    if not top_products:
        top_products = recommendations_list

    product_names = [product.name for product in top_products]
    product_ratings = [getattr(product, 'average_rating', 0) for product in top_products]  # Fallback if no rating

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(product_names, product_ratings, color='skyblue')
    ax.set_xlabel('Average Rating')
    ax.set_title('Top Recommended Products')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    bar_chart_img = base64.b64encode(img_buf.getvalue()).decode('utf-8')

    return render(request, 'dashboard.html', {
        'recommendations': recommendations_list,
        'pie_chart_img': pie_chart_img,
        'bar_chart_img': bar_chart_img,
    })