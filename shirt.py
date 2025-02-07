import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
import torch

def get_dominant_color(image, roi):
    # Extract the region of interest (shirt area)
    x, y, w, h = roi
    shirt_region = image[y:y+h, x:x+w]
    
    # Reshape the image for K-means clustering
    pixels = shirt_region.reshape(-1, 3)
    
    # Perform K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    
    # Get the dominant color (the cluster center with most points)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = kmeans.cluster_centers_[counts.argmax()]
    
    return dominant_color

def color_name(rgb):
    # Dictionary of basic colors and their RGB values
    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'white': [255, 255, 255],
        'black': [0, 0, 0],
        'yellow': [255, 255, 0],
        'purple': [128, 0, 128],
        'orange': [255, 165, 0],
        'brown': [165, 42, 42],
        'pink': [255, 192, 203],
        'gray': [128, 128, 128]
    }
    
    # Find the closest color by minimum Euclidean distance
    min_dist = float('inf')
    closest_color = None
    
    for color_name, color_rgb in colors.items():
        dist = np.sqrt(sum((rgb - color_rgb) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    
    return closest_color

def main():
    st.title("Shirt Color Detection App")
    st.write("Upload an image to detect the color of the shirt")
    
    # Load YOLO model
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt')
    
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image = np.array(pil_image)
        
        # Run inference
        results = model(image)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a person
                if result.names[int(box.cls)] == 'person':
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Define shirt region (approximate)
                    shirt_x = x1
                    shirt_y = y1 + (y2 - y1) // 3  # Start from upper body
                    shirt_w = x2 - x1
                    shirt_h = (y2 - y1) // 3  # Take middle third of body
                    
                    # Get dominant color
                    dominant_rgb = get_dominant_color(image, (shirt_x, shirt_y, shirt_w, shirt_h))
                    color = color_name(dominant_rgb)
                    
                    # Draw rectangle around shirt area
                    cv2.rectangle(image, 
                                (shirt_x, shirt_y), 
                                (shirt_x + shirt_w, shirt_y + shirt_h),
                                (0, 255, 0), 2)
                    
                    # Display detected color
                    st.write(f"Detected shirt color: {color}")
        
        # Display the processed image
        st.image(image, caption='Processed Image', use_column_width=True)

if __name__ == '__main__':
    main()