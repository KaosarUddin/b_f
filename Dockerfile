# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Upgrade pip and install your package
RUN pip install --upgrade pip
RUN pip install .

# Set the default command (optional: show CLI help when container runs)
CMD ["spd-id", "--help"]
