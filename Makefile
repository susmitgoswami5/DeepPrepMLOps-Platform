.PHONY: help install up down demo test retrain evaluate

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
	python3 -m src.data.mock_generator
	python3 -m src.monitoring.drift_detector
	@echo "🎉 Demo pipeline simulation complete. Check UI for updates."

retrain: ## Manually trigger the Model Retraining Pipeline
	python3 scripts/retrain.py
	python3 scripts/validate_model.py
	@echo "✅ Retraining and Validation Gate executed."

test: ## Run unit tests
	python3 -m pytest -q tests/

evaluate: ## Re-run the held-out evaluation and ablation (writes evaluation/results.md)
	python3 -m evaluation.evaluate

format: ## Format code with black/flake8 (if installed)
	black src/
	flake8 src/
