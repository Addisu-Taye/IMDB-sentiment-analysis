# wsgi.py
import sys
import os

# Add the project root to the sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import your Flask app instance from app.py
# Vercel's server will look for a variable named 'application' by default
from app import app as application