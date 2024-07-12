# api/serializers.py

from rest_framework import serializers

class PredictSerializer(serializers.Serializer):
    image = serializers.CharField()
