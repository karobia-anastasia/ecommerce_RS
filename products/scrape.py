import logging
import requests
from bs4 import BeautifulSoup
from .models import Product
from requests.exceptions import RequestException

# Set up a logger for this module
logger = logging.getLogger(__name__)

def scrape_product(url):
    """
    Scrapes product details from the given URL and stores them in the database.
    Logs information and errors along the way.
    """
    logger.info(f"Attempting to scrape product from: {url}")  # Log the URL being scraped

    try:
        # Send a GET request to the product page
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx/5xx)

        # Parse the page content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all product containers on the page (adjust the selector as needed)
        products_on_page = soup.find_all('div', {'class': 'product-container'})  # Adjust as needed
        
        if not products_on_page:
            logger.warning(f"No products found on page: {url}")
            return

        # Iterate over each product container and extract details
        for product_elem in products_on_page:
            try:
                product_name = product_elem.find('h2', {'class': 'product-title'}).text.strip()  # Adjust as needed
                product_price = product_elem.find('span', {'class': 'price'}).text.strip()  # Adjust as needed
                product_description = product_elem.find('p', {'class': 'product-description'}).text.strip()  # Adjust as needed
                product_image_url = product_elem.find('img')['src'].strip()  # Adjust as needed

                # Log extracted details
                logger.info(f"Product Name: {product_name}")
                logger.info(f"Product Price: {product_price}")
                logger.info(f"Product Description: {product_description}")
                logger.info(f"Product Image URL: {product_image_url}")

                # Save or update the product in the database
                product, created = Product.objects.update_or_create(
                    name=product_name,
                    defaults={
                        'price': product_price,
                        'description': product_description,
                        'image_url': product_image_url,
                    }
                )

                # Log the result of the database operation
                logger.info(f"Product {'created' if created else 'updated'}: {product_name}")
            
            except AttributeError as e:
                logger.error(f"Error extracting product details: {e} - Skipping this product.")
                continue  # Skip this product if there's any missing data

    except RequestException as e:
        logger.error(f"Error scraping product from {url}: {e}")
    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {e}")
