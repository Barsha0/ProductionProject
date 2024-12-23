"""
WSGI config for LinkSafe project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

import sys
import io

# Set UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'LinkSafe.settings')

application = get_wsgi_application()
