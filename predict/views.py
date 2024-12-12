from django.shortcuts import render
from django.core.files.storage import default_storage
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Import the form
from .forms import ImageUploadForm

# Load your trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'my_model.keras')
model = load_model(model_path)

def predict_view(request):
    result = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            uploaded_image = request.FILES['image']
            file_path = default_storage.save(f"uploads/{uploaded_image.name}", uploaded_image)

            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))  # Adjust size as per your model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize if required by your model

            # Make a prediction
            predictions = model.predict(img_array)
            result = np.argmax(predictions, axis=1)[0]  # Example for classification

    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form, 'result': result})
