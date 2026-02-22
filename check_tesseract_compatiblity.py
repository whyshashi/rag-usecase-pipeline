import pytesseract

# This should print the Tesseract version if Python can find it
try:
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    print(f"Error finding Tesseract: {e}")