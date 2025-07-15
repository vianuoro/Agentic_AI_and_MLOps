# Makefile for house_price_agent

PROJECT_DIR=house_price_agent
PYTHON=python

# Run the FastAPI server
serve:
	uvicorn $(PROJECT_DIR)/api.app:app --reload --host 0.0.0.0 --port 8000

# Train the model (assumes ml/train.py has a main entry)
train:
	$(PYTHON) $(PROJECT_DIR)/ml/train.py

# Evaluate the model
evaluate:
	$(PYTHON) $(PROJECT_DIR)/ml/evaluate.py

# Run prediction (example script or manually test API)
predict:
	$(PYTHON) $(PROJECT_DIR)/ml/predict.py

# Run drift check
drift-check:
	$(PYTHON) $(PROJECT_DIR)/monitoring/drift_check.py

# Run pipeline DAG
pipeline:
	$(PYTHON) $(PROJECT_DIR)/dags/data_pipeline.py

# Run tests (placeholder)
test:
	echo "No tests yet. Add unit tests to a tests/ folder."

# Build Docker image
docker-build:
	docker build -t house-price-agent .

# Run Docker container
docker-run:
	docker run -p 8000:8000 house-price-agent

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -name "*.pyc" -delete

# Run DVC pipeline
dvc-run:
	dvc repro

.PHONY: serve train evaluate predict drift-check pipeline test docker-build docker-run clean dvc-run
