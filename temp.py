import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
import os
from sklearn.neighbors import NearestNeighbors
import PIL
from PIL import Image

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load the feature embeddings and filenames
embeddings = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filename.pkl', 'rb'))

# Define the function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    features = model.predict(tf.expand_dims(x, axis=0)).flatten()
    normalized_features = features / norm(features)
    return normalized_features

# Define the function to find similar images
def find_similar_images(image_path, embeddings, filenames, model, n=5):
    query_features = extract_features(image_path, model)
    neighbors = NearestNeighbors(n_neighbors=n, algorithm='brute', metric='euclidean')
    neighbors.fit(embeddings)
    distances, indices = neighbors.kneighbors([query_features])
    similar_images = []
    for index in indices[0]:
        similar_images.append(filenames[index])
    return similar_images

# Create the Streamlit app
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Find Similar Fashion Images')
st.write('Upload an image and find similar fashion images.')

# Add the file uploader
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

# If the user uploaded a file, find similar images and display them
if uploaded_file is not None:
    # Save the uploaded file to disk
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    # Find similar images
    similar_images = find_similar_images(uploaded_file.name, embeddings, filenames, model)
    # Display the similar images
    for image_path in similar_images:
        image = Image.open(image_path)
        st.image(image, caption=image_path)


