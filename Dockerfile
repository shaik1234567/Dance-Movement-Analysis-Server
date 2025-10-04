# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the file that lists the dependencies
COPY requirements.txt .

# Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install the Python dependencies from the requirements file
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Copy all your project files (app.py, analyzer.py, etc.) into the container
COPY . .

# Tell Docker that the container will listen on port 5000
EXPOSE 5000

# The command to run when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]

COPY . /app
WORKDIR /app
RUN pip install pytest
