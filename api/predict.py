# api/predict.py

import numpy as np
import cv2
import base64
import io
from PIL import Image

# Load your model weights and biases here
loaded_model = np.load('D:\Projects\hand-digit-fullstack\hand_written_digit_recognitio\hand-written-digit-recognition\Backend\model\my_model.npz')
weights_input_hidden = loaded_model['weights_input_hidden']
biases_hidden = loaded_model['biases_hidden']
weights_hidden_output = loaded_model['weights_hidden_output']
biases_output = loaded_model['biases_output']

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)



def predict_digit(image_data):
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = np.array(image)
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = thresh_image[y:y+h, x:x+w]
    else:
        cropped_image = thresh_image
    
    # Invert colors and resize to 28x28
    inverted_image = 255 - cropped_image
    resized_image = cv2.resize(inverted_image, (28, 28))
    
    # Normalize the image
    normalized_image = resized_image / 255.0
    final_image = 1 - normalized_image
    
    # Flatten the image
    flattened_image = final_image.reshape(1, -1)
    
    input_layer = flattened_image
    hidden_layer_input = np.dot(input_layer, weights_input_hidden) + biases_hidden
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))  # Sigmoid activation
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    output_layer_output = softmax(output_layer_input)
    prediction = np.argmax(output_layer_output)
    
    return int(prediction)
