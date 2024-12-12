from django.shortcuts import render
from .forms import ImageUploadForm
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

# Load the trained model (make sure the path is correct)
model = tf.keras.models.load_model('my_model.keras')  # Adjust with your model path

def predict_view(request):
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        
        # Save the uploaded image temporarily
        img_path = os.path.join('uploads', img.name)  # You can use a directory like 'uploads'
        with open(img_path, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # Load and preprocess the image for prediction
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust to your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image (if your model was trained with normalized data)

        # Predict using the trained model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)  # Get the index of the highest probability

        # Map the predicted class to A, B, C, or D
        if predicted_class == 0:
            result = 'A'
        elif predicted_class == 1:
            result = 'B'
        elif predicted_class == 2:
            result = 'C'
        elif predicted_class == 3:
            result = 'D'

    # Render the result and form
    return render(request, 'index.html', {'result': result, 'form': ImageUploadForm()})
