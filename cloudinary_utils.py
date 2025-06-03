import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET"),
)

def upload_to_cloudinary(filepath):
    response = cloudinary.uploader.upload(filepath)
    return response.get("secure_url")
