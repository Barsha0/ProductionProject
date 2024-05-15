from django.db import models

# Create your models here.

class Links(models.Model):
    link_str = models.CharField(max_length=30)
