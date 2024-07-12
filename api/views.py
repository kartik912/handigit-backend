# # api/views.py

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .predict import predict_digit
# from .serializers import PredictSerializer
# from django.http import HttpResponse, JsonResponse
# from datetime import datetime, timedelta
# # from django.db import connection
# # from joblib import load
# # import pickle
# from django.views.decorators.csrf import csrf_exempt
# # # Create your views here.
# import json
# # from bson import ObjectId

# # from .models import users_collection

# # from decouple import config

# # import bcrypt
# # import hashlib
# # import jwt

# class PredictView(APIView):
#     def post(self, request):
#         serializer = PredictSerializer(data=request.data)
#         print(request.data)
#         if serializer.is_valid():
#             image_data = serializer.validated_data['image']
#             prediction = predict_digit(image_data)
#             return Response({'prediction': prediction}, status=status.HTTP_200_OK)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# @csrf_exempt
# def Index(request):
#     return HttpResponse('<h1>This is the personality predictor backend server.</h1>')


# @csrf_exempt
# def ml(request):
#     if request.method == 'POST':
#         image_data = json.loads(request.body)['image']
#         # data = request.POST.get('image')  # Assuming 'image' is the key for your image data
#         print(image_data)
        
#         return JsonResponse({'result': 'success'})  # Replace with your actual response
    
#     return JsonResponse({'error': 'POST method required'})

import base64
import io
import json
import numpy as np
from PIL import Image
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import os
model_path = os.path.join(os.path.dirname(__file__), 'model', 'my_model.npz')
loaded_model = np.load(model_path)
weights_input_hidden = loaded_model['weights_input_hidden']
biases_hidden = loaded_model['biases_hidden']
weights_hidden_output = loaded_model['weights_hidden_output']
biases_output = loaded_model['biases_output']

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        try:
            # Get image data from request
            image_data = json.loads(request.body)['image']
            # image_file = request.FILES['image']
            # Decode the base64 image data
            
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data)).convert('L')
            image = np.array(image)
            _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            inverted_image = 255 - thresh_image
            resized_image = cv2.resize(inverted_image, (28, 28))
            normalized_image = resized_image / 255.0
            final_image = normalized_image
            # flattened_image = final_image.reshape(1, -1)
            flattened_image = final_image.flatten().tolist()

            # Perform prediction
            input_layer = np.array([flattened_image])
            hidden_layer_input = np.dot(input_layer, weights_input_hidden) + biases_hidden
            hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))  # Sigmoid activation
            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
            output_layer_output = softmax(output_layer_input)
            prediction = np.argmax(output_layer_output)
            prediction = int(prediction)
            print(image)

            return JsonResponse({'prediction': prediction})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def Index(request):
    return HttpResponse('<h1>This is the personality predictor backend server.</h1>')