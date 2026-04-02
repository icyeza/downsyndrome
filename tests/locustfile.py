"""
Locust load testing script for the Down Syndrome Classification API
Tests API performance under load
"""

from locust import HttpUser, task, between, LoadTestShape
import random
import os
import io
import json
from PIL import Image


def make_random_image_bytes():
    img = Image.new('RGB', (224, 224), color=(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


class APILoadTest(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.client.timeout = 30
        # Cache available model IDs for switch-model tests
        self._model_ids = []
        resp = self.client.get("/models")
        if resp.status_code == 200:
            self._model_ids = [m['id'] for m in resp.json()]

    # ── Read-only endpoints ────────────────────────────────────────────────

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def get_model_info(self):
        self.client.get("/info")

    @task(1)
    def get_stats(self):
        self.client.get("/stats")

    @task(2)
    def list_models(self):
        self.client.get("/models")

    @task(1)
    def get_retrain_status(self):
        self.client.get("/retrain-status")

    @task(1)
    def get_prediction_history(self):
        limit = random.choice([10, 50, 100])
        self.client.get(f"/prediction-history?limit={limit}")

    # ── Prediction endpoints ───────────────────────────────────────────────

    @task(5)
    def predict_image(self):
        try:
            files = {'file': ('test.jpg', make_random_image_bytes(), 'image/jpeg')}
            self.client.post("/predict", files=files)
        except Exception as e:
            print(f"Error in predict_image: {e}")

    @task(2)
    def predict_batch(self):
        """Send 2-4 images as a batch."""
        try:
            batch_size = random.randint(2, 4)
            files = [
                ('files', (f'img_{i}.jpg', make_random_image_bytes(), 'image/jpeg'))
                for i in range(batch_size)
            ]
            self.client.post("/predict-batch", files=files)
        except Exception as e:
            print(f"Error in predict_batch: {e}")

    # ── Training / management endpoints ───────────────────────────────────

    @task(1)
    def upload_training_data(self):
        """Upload a single image as labelled training data."""
        try:
            label = random.choice(['downSyndrome', 'noDownSyndrome'])
            files = {'files': ('train.jpg', make_random_image_bytes(), 'image/jpeg')}
            data = {'label': label}
            self.client.post("/upload-training-data", files=files, data=data)
        except Exception as e:
            print(f"Error in upload_training_data: {e}")

    @task(1)
    def upload_predict_mode(self):
        """Upload images in bulk-predict mode (label=predict)."""
        try:
            files = {'files': ('test.jpg', make_random_image_bytes(), 'image/jpeg')}
            data = {'label': 'predict'}
            self.client.post("/upload-training-data", files=files, data=data)
        except Exception as e:
            print(f"Error in upload_predict_mode: {e}")

    @task(1)
    def switch_model(self):
        """Switch to a random registered model (read from /models on start)."""
        if not self._model_ids:
            return
        model_id = random.choice(self._model_ids)
        self.client.post(
            "/switch-model",
            json={"model_id": model_id},
            headers={"Content-Type": "application/json"},
        )

    # Retrain is intentionally low-weight — it's expensive and stateful.
    # Weight 0 here; uncomment and increase if you want retrain load tests.
    # @task(1)
    # def trigger_retrain(self):
    #     params = {"epochs": 1, "learning_rate": 0.0001, "batch_size": 32, "optimizer": "adam"}
    #     self.client.post("/retrain", json=params)


class StepLoadShape(LoadTestShape):
    """
    Increases load step by step.
    """

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 60:
            return (10, 1)   # 10 users, spawn rate 1
        elif run_time < 120:
            return (25, 1)
        elif run_time < 180:
            return (50, 1)
        elif run_time < 240:
            return (100, 2)
        else:
            return None      # Stop test


if __name__ == "__main__":
    print("Locust load testing module loaded")
    print("\nTo run tests:")
    print("  locust -f tests/locustfile.py -H http://localhost:5000 --headless -u 100 -r 10 -t 5m")
    print("\nOr with web UI:")
    print("  locust -f tests/locustfile.py -H http://localhost:5000")
