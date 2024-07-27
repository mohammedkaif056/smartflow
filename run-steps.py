# Imports
import os
from configparser import ConfigParser

# Variables
python_packages = ["fastapi", "uvicorn", "jinja2", "opencv-python", "imutils", "pytesseract", "numpy", "Pillow", "mysql-connector-python"]

# Initializing the "ConfigParser" Class
config_parser_object = ConfigParser()

# Setting the Data in the "config_parser_object" Variable
config_parser_object["MYSQL"] = {
    "password": str(input("\nEnter Your Database Password: "))
}

# Writing "config_parser_object" to the "config.ini" File
with open("MySQL/config.ini", "w") as config_file:
    config_parser_object.write(config_file)

# Start Text
print("\nInstalling & Upgrading The Necessary Packages and Tools Required For Smart Flow.\n")

# Installing and Upgrading PIP + Python Packages
os.system("python -m pip install --upgrade pip")
os.system("pip install --upgrade {0}".format(" ".join(python_packages)))