from django.urls import path
from .views import link_checker


urlpatterns = [
        path('checker/', link_checker, name='link-checker'),
        ]

