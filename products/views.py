from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from products.form import ProductForm
from products.recommendation import *
from .models import Order, Product, Transaction, Cart
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import numpy as np
import logging
from django.shortcuts import render, redirect
from .models import Product
import seaborn as sns
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
import base64
from .models import Cart, Transaction, Product
from django.shortcuts import render, redirect
from products.recommendation import *
from .models import Product
import logging


# Set up logging
logger = logging.getLogger(__name__)
import logging

# Set up logging
logger = logging.getLogger(__name__)




import logging
import requests
from bs4 import BeautifulSoup
from .models import Product
from requests.exceptions import RequestException
logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
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

# Set up logger for better debugging
logger = logging.getLogger(__name__)

@login_required



# Product Create (Create new product)
def product_create(request):
 
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')  # Redirect to the product list after saving
    else:
        form = ProductForm()

    return render(request, 'product_form.html', {'form': form})





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


# Set up logging
logger = logging.getLogger(__name__)

def product_list(request):
    """
    Displays a list of products along with recommended products based on KNN and fallback to popular products.
    """
    if not request.user.is_authenticated:
        return redirect('login')  # Redirect to login page if user is not authenticated

    try:
        # Get the current user ID
        user_id = request.user.id  # Assuming you have a logged-in user

        # Get the product list from the database
        products = Product.objects.all().order_by('-id')  # Show most recently added products first

        # Get combined recommendations (KNN or popular products)
        recommended_products = get_combined_recommendations(user_id, top_n=5)

        # Render the template and pass the products and recommendations
        context = {
            'products': products,
            'recommended_products': recommended_products
        }

    except Exception as e:
        # Log the exception details for debugging
        logger.error(f"Error occurred while fetching products or recommendations: {e}", exc_info=True)

        # Provide user-friendly feedback if there’s an error
        context = {
            'products': Product.objects.all(),
            'recommended_products': [],
            'error_message': 'An error occurred while fetching the products and recommendations. Please try again later.'
        }

    return render(request, 'product_list.html', context)

@login_required
def dashboard(request):
    """ Renders the dashboard with product recommendations and visualizations """
    try:
        # Get the user ID (the user is guaranteed to be logged in due to the @login_required decorator)
        user_id = request.user.id

        # Step 1: Get KNN-based recommendations (or fallback to popular products)
        recommended_products = get_combined_recommendations(user_id, top_n=5)

        # Step 2: Create or update the user-item matrix and generate the heatmap
        user_item_matrix, user_item_matrix_plot = create_user_item_matrix()

        # Step 3: Generate a plot of popular products (if any)
        popular_products_plot = plot_popular_products(top_n=5)

        # Step 4: Check for updates in the user's cart or transactions
        if 'update_cart' in request.POST:
            product_id = request.POST.get('product_id')
            interaction_type = request.POST.get('interaction_type', 'cart')  # Default to 'cart' if not provided
            # Trigger the real-time update after cart interaction (add/update/remove)
            updated_recommendations = on_cart_update(user_id, product_id, interaction_type=interaction_type, top_n=5)
            # Get updated recommendations (fallback to popular products if necessary)
            recommended_products = get_combined_recommendations(user_id, top_n=5)

        # Step 5: Prepare context for rendering
        context = {
            'recommended_products': recommended_products,
            'user_item_matrix': user_item_matrix,  # Pass the user-item matrix if needed
            'user_item_matrix_plot': user_item_matrix_plot,
            'popular_products_plot': popular_products_plot,
            'user_id': user_id,  # Pass the user_id to the template for user-specific content
        }

        return render(request, 'dashboard.html', context)

    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return render(request, 'dashboard.html', {'error': 'An error occurred while generating recommendations.'})

# Helper function to update recommendations after cart interaction (if required)
def on_cart_update(user_id, product_id, interaction_type='cart', top_n=5):
    """
    Updates recommendations after the user updates their cart.
    """
    try:
        # Placeholder for real-time cart update logic
        # After the cart is updated, generate new recommendations based on the updated state
        logger.info(f"User {user_id} updated their {interaction_type} with product {product_id}.")
        updated_recommendations = get_combined_recommendations(user_id, top_n=top_n)
        return updated_recommendations
    except Exception as e:
        logger.error(f"Error updating cart recommendations for user {user_id}: {e}")
        return []  # Return empty list if an error occurs
