import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    """
    Load the image from a file and return it as a NumPy array.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - np.array: The image as a NumPy array.
    """
    # Open the image using Pillow and convert it to RGB mode
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    image_np = np.array(image)
    
    return image_np

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    """
    Compress the image by reducing the number of colors using KMeans clustering.

    Parameters:
    - image_np (np.array): The image as a NumPy array (H x W x C).
    - n_colors (int): The number of colors to reduce the image to.

    Returns:
    - np.array: The compressed image as a NumPy array.
    """
    # Get the dimensions of the image
    h, w, c = image_np.shape
    
    # Reshape the image to a 2D array of pixels (each pixel is a row with 3 columns for RGB)
    pixels = image_np.reshape(-1, 3)
    
    # Apply KMeans clustering to the pixel data
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers (these will be the new colors)
    new_colors = kmeans.cluster_centers_.astype('uint8')
    
    # Map each pixel to its nearest cluster center
    labels = kmeans.labels_
    
    # Replace each pixel with its corresponding new color
    compressed_image_np = new_colors[labels].reshape(h, w, c)
    
    return compressed_image_np

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = 'favorite_image.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
