# Use a Python base image (version 3.10 or higher)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies (including Git & SSH)
RUN apt update && apt install -y git openssh-client

# Copy the contents of the current directory (your code) into the container
COPY . .

# Install any dependencies if you have a requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your model
CMD ["bash"]