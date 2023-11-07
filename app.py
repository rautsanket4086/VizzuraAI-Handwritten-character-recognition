import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import characterModel

# Page title and description
st.title("Handwritten Text Classification")
st.write("Welcome to the handwritten text classification project.")
st.write("Learn about machine learning while using the interface.")

# Load MNIST data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')

# Split data                                                            
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Function to make predictions
def preprocess_image(image):
    # Open the image using PIL
    img = Image.open(image)

    # Resize the image to the desired dimensions (e.g., 28x28)
    img = img.resize((28, 28))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values (if needed)
    img_array = img_array / 255.0

    # Convert to a PyTorch tensor
    img_tensor = torch.tensor(img_array)

    return img_tensor

# Load your trained model here
model = characterModel()  # Initialize your model
model.load_state_dict(torch.load('characterModel.pth'))  # Load the trained model weights
model.eval()  # Set the model to evaluation mode                                                                                                                                                                                                                                                                                                                                                                                

# Function to make predictions using the loaded model
def predict(image):
    image = preprocess_image(image)

    # Convert the image to a PyTorch tensor
    img_tensor = image.view(1, 784).float()

    # Make predictions using your loaded model
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction).item()

    return predicted_class

# Educational content
st.subheader("Learn about Machine Learning")
st.write("This interface helps you classify handwritten digits using machine learning.")
st.write("Machine learning is a type of artificial intelligence (AI) that allows computers to learn and make decisions without being explicitly programmed.")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Classification button
    if st.button("Classify"):
        # Call the predict function
        prediction = predict(uploaded_image)

        # Display the result
        st.write(f"Predicted Class: {prediction}")

# Display some training examples (you can customize this)
st.subheader("Training Examples")
for i in range(10):
    st.image(X_train[i].reshape(28, 28), caption=f"Label: {y_train[i]}", use_column_width=True)

# Machine learning explanation
st.subheader("How Does Machine Learning Work?")
st.write("Machine learning models learn from data and make predictions or decisions. Here's a simplified explanation:")
st.write("1. Data Collection: We collect a dataset with input features (images of digits) and target labels (the digit's value).")
st.write("2. Model Training: The machine learning model learns from the dataset to understand the patterns between images and labels.")
st.write("3. Prediction: Once trained, the model can predict the label of a new image.")
st.write("In this project, the model has learned to recognize handwritten digits.")

# Data visualization (customize this part)
st.subheader("Data Visualization")
st.write("Visualization is an essential part of understanding data and model results.")
if st.button("Show Plot"):
    # Generate sample data for plotting (you can replace this with your data)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create an interactive plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

# Model evaluation (customize this part)
st.subheader("Model Evaluation")
st.write("Evaluating the model helps us understand how well it performs.")
if st.button("Evaluate Model"):
    # Evaluate your model (e.g., calculate accuracy)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Add sliders for customization
st.sidebar.title("Customization")
slider_value = st.sidebar.slider("Select a value", 0, 100, 50)
st.write(f"Selected Value: {slider_value}")

# Add buttons for customization
if st.button("Click Me"):
    st.write("Button Clicked!")

