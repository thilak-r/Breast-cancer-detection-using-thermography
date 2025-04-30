import os
from flask import Flask, request, jsonify, render_template
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNN_Model

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model().to(device)
model.load_state_dict(torch.load('breast_cancer_detection_model.pth'))
model.eval()  # Set the model to evaluation mode
print(model)


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((20, 15)),  # Resize to match expected dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open the image and apply the transformation
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform prediction
        with torch.no_grad():
            output = model(img).squeeze()  # Get the prediction output
            prediction = (output > 0.5).item()  # Binary classification (0 or 1)

        # Return the prediction result
        result = 'Sick' if prediction >= 0.5 else 'Normal'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to check if the server is running
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
