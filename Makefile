.PHONY: help install up down demo test retrain

help: ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies locally (Virtual Environment)
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt

up: ## Start the entire MLOps stack via Docker Compose
	docker-compose up -d --build
	@echo "🚀 API is running at http://localhost:8000"
	@echo "📊 Dashboard is running at http://localhost:8501"

down: ## Tear down the Docker Compose stack
	docker-compose down -v

demo: ## Generate mock logs and run full pipeline simulation
	python3 data/mock_generator.py
	python3 src/monitoring/drift_detector.py
	@echo "🎉 Demo pipeline simulation complete. Check UI for updates."

retrain: ## Manually trigger the Model Retraining Pipeline
	python3 scripts/retrain.py
	python3 scripts/validate_model.py
	@echo "✅ Retraining and Validation Gate executed."

test: ## Run unit tests (pytest must be installed)
	python3 -m pytest tests/

format: ## Format code with black/flake8 (if installed)
	black src/
	flake8 src/
