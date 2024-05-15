from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import LinkSerializer
from .models import Links



class LinkViewset(viewsets.ModelViewSet):
    model = Links
    queryset = Links.objects.all()
    serializer_class = LinkSerializer
    


@api_view(['post'])
def link_checker(request):
    if request.data.get('url'):
        return Response("Link is Valid")
    else:
        return Response("Link is invalid")

