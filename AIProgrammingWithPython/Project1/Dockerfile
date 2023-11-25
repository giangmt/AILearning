# Use a base operating system like Ubuntu
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install libraries from requirements.txt
RUN pip install -r requirements.txt

# Copy the entire contents of the project directory into the container
COPY . .

# Run the application or the default command when the container starts
CMD ["python", "check_images.py"]
