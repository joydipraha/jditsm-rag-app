# Use the full Python image, which includes necessary build tools (like gcc) 
# to compile packages like scikit-learn from requirements.txt.
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies using pip
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run expects the application to listen on the port defined by the PORT environment variable.
ENV PORT 8080

# Command to run the application using gunicorn (assuming your main app is in app.py)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app
