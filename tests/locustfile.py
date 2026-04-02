"""
Locust load testing script for the Down Syndrome Classification API
Tests API performance under load
"""

from locust import HttpUser, task, between, LoadTestShape
import random
import os
import io
from PIL import Image


class APILoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before tests start"""
        self.client.timeout = 30
    
    @task(3)
    def health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")
    
    @task(5)
    def predict_image(self):
        """Test prediction with random image"""
        try:
            # Create random image
            img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), 
                                                       random.randint(0, 255), 
                                                       random.randint(0, 255)))
            
            # Save to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Make request
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            self.client.post("/predict", files=files)
        
        except Exception as e:
            print(f"Error in prediction test: {e}")
    
    @task(2)
    def get_model_info(self):
        """Test model info endpoint"""
        self.client.get("/info")
    
    @task(1)
    def get_stats(self):
        """Test stats endpoint"""
        self.client.get("/stats")


class StepLoadShape(LoadTestShape):
    """
    A load shape that increases load step by step
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 1)  # 10 users, 1 spawn rate
        elif run_time < 120:
            return (25, 1)
        elif run_time < 180:
            return (50, 1)
        elif run_time < 240:
            return (100, 2)
        else:
            return None  # Stop test


if __name__ == "__main__":
    print("Locust load testing module loaded")
    print("\nTo run tests:")
    print("  locust -f tests/locustfile.py -H http://localhost:5000 --headless -u 100 -r 10 -t 5m")
    print("\nOr with web UI:")
    print("  locust -f tests/locustfile.py -H http://localhost:5000")
