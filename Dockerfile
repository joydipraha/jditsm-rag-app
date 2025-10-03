# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
# This means dependencies are only reinstalled if requirements.txt changes
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the app will listen on (required by Cloud Run)
EXPOSE 8080

# Run the application using Gunicorn, a production-ready WSGI server
# The Gunicorn command starts the app on port 8080 and sets the number of workers.
# The --timeout 600 flag increases the timeout to 10 minutes to allow
# for the initial data and embeddings loading on cold starts.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "app:app"]
