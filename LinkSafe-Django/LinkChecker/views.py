from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import LinkSerializer
from .models import Links            
from Model import preprocess_url
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib



class LinkViewset(viewsets.ModelViewSet):
    model = Links
    queryset = Links.objects.all()
    serializer_class = LinkSerializer
    
    # Load the scaler used for preprocessing
    scaler = joblib.load('scaler.save')


    # Load the saved model
    trained_model = tf.keras.models.load_model('C:\\Users\\user\\OneDrive\\Desktop\\Production\\LinkSafe-Django\\CNNmodel.h5')


@api_view(['post'])
def link_checker(request):
        if request.data.get('url'):
            # Preprocess the URL for prediction
            preprocessed_url = preprocess_url(request.data['url'], LinkViewset.scaler)
            
            # Make prediction
            prediction = LinkViewset.trained_model.predict(preprocessed_url)
            binary_prediction = (prediction > 0.5).astype(int)
            
            # Return the prediction result
            if binary_prediction[0][0] == 1:
                return Response({"message": "The URL is predicted to be malicious."})
            else:
                return Response({"message": "The URL is predicted to be genuine."})
        else:
            return Response({"error": "No URL provided."})
    



