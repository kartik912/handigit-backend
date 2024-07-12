# api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.Index),
    path('predict', views.predict_image, name='predict'),
]
