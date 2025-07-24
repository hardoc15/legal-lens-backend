# Use a base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port used by Flask (or Gunicorn)
EXPOSE 8080

# Start the app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "src.api:app"]
