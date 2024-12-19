from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import LinkSerializer
from .models import Links
from Model import isSafeUrl



class LinkViewset(viewsets.ModelViewSet):
    model = Links
    queryset = Links.objects.all()
    serializer_class = LinkSerializer



@api_view(['post'])
def link_checker(request):
    url = request.data.get('url')
    if isSafeUrl(url):
        return Response({"isSafeUrl" : True, "message" : "The url is safe"})
    else:
        return Response({"isSafeUrl" : False, "message" : "The url is unsafe"})


