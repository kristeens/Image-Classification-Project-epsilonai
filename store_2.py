
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import cv2

# Set page configuration
st.set_page_config(
    page_title="Fast Food Classifier",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #FF6B35 0%, #FF8E35 100%);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Fast food classes (based on common fast food categories)
CLASS_NAMES = [
    'Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito'
]

class FastFoodClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(FastFoodClassifier, self).__init__()
        self.model = models.DenseNet201(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    model = FastFoodClassifier(num_classes=len(CLASS_NAMES))
    
    # In a real scenario, you would load your trained weights here
    # For demo purposes, we'll use a randomly initialized model
    # model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor):
    """Make prediction on the image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item(), probabilities.numpy()[0]

def plot_predictions(probabilities, class_names):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort probabilities and get top 5
    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    bars = ax.barh(top_classes, top_probs, color='#FF6B35')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Top 5 Predictions', fontsize=14)
    ax.set_xlim(0, 1)
    
    # Add probability values on bars
    for bar, prob in zip(bars, top_probs):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🍔 Fast Food Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Classify your fast food images with AI!")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a deep learning model to classify fast food items "
        "into 20 different categories including burgers, pizza, fries, and more!"
    )
    
    st.sidebar.title("Model Information")
    st.sidebar.text("Architecture: ResNet-50")
    st.sidebar.text(f"Classes: {len(CLASS_NAMES)}")
    st.sidebar.text("Input Size: 224x224")
    
    st.sidebar.title("Instructions")
    st.sidebar.text("1. Upload a fast food image")
    st.sidebar.text("2. Or take a picture")
    st.sidebar.text("3. View predictions")
    
    # Load model
    model = load_model("D:\Coures\Machinelearning\epsilon_ai\Deep_Learning\final\final_fastfood_DenseNet.h5")
    
    # Image input methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fast food image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of fast food"
        )
    
    with col2:
        st.subheader("Or Take a Picture")
        camera_image = st.camera_input("Take a picture of your fast food")
    
    # Use either uploaded file or camera image
    image = uploaded_file if uploaded_file is not None else camera_image
    
    if image is not None:
        # Display original image
        original_image = Image.open(image)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(original_image, use_column_width=True)
            
            # Image information
            st.write(f"**Image Size:** {original_image.size}")
            st.write(f"**Image Mode:** {original_image.mode}")
        
        with col2:
            # Make prediction
            processed_image = preprocess_image(original_image)
            predicted_class, confidence, probabilities = predict_image(model, processed_image)
            
            # Display prediction results
            st.subheader("🔍 Prediction Results")
            
            # Main prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Class: {CLASS_NAMES[predicted_class].replace('_', ' ').title()}</h3>
                <p><strong>Confidence:</strong> {confidence:.3f}</p>
                <div class="confidence-bar" style="width: {confidence * 100}%"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.subheader("📊 Prediction Probabilities")
            fig = plot_predictions(probabilities, CLASS_NAMES)
            st.pyplot(fig)
        
        # Additional features
        st.markdown("---")
        st.subheader("📈 Additional Analysis")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            # Confidence score
            confidence_percent = confidence * 100
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence_percent:.1f}%"
            )
        
        with col4:
            # Class index
            st.metric(
                label="Class Index",
                value=predicted_class
            )
        
        with col5:
            # Total classes
            st.metric(
                label="Total Classes",
                value=len(CLASS_NAMES)
            )
        
        # Show all classes with probabilities
        if st.checkbox("Show all class probabilities"):
            st.subheader("All Class Probabilities")
            
            # Create a DataFrame-like display
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                col6, col7, col8 = st.columns([2, 6, 2])
                with col6:
                    st.text(f"{i+1}.")
                with col7:
                    st.text(class_name.replace('_', ' ').title())
                with col8:
                    st.text(f"{prob:.3f}")
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        results_text = f"""
        Fast Food Classification Results:
        - Predicted Class: {CLASS_NAMES[predicted_class]}
        - Confidence: {confidence:.3f}
        - Top 5 Predictions:
        """
        
        top_indices = np.argsort(probabilities)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            results_text += f"\n  {i+1}. {CLASS_NAMES[idx]}: {probabilities[idx]:.3f}"
        
        st.download_button(
            label="Download Results as Text",
            data=results_text,
            file_name="fast_food_classification_results.txt",
            mime="text/plain"
        )

    else:
        # Show sample images when no image is uploaded
        st.info("👆 Please upload an image or take a picture to get started!")
        
        # Display sample categories
        st.subheader("🍽️ Supported Fast Food Categories")
        
        categories_per_row = 5
        for i in range(0, len(CLASS_NAMES), categories_per_row):
            cols = st.columns(categories_per_row)
            for j, class_name in enumerate(CLASS_NAMES[i:i+categories_per_row]):
                with cols[j]:
                    st.info(f"**{class_name.replace('_', ' ').title()}**")

if __name__ == "__main__":
    main()
