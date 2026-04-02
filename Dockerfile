# Docker configuration for Down Syndrome Classification API
# Enables containerized deployment

FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=api/app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "api/app.py"]
