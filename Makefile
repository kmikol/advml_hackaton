PYTHON := python3

# Default target
all: run

run:
	$(PYTHON) src/advml_hackaton/environmental/data_preprocessing.py && \
	$(PYTHON) src/advml_hackaton/environmental/hyperparameter_optim.py && \
	$(PYTHON) src/advml_hackaton/environmental/select_best_model.py && \
	$(PYTHON) src/advml_hackaton/environmental/train.py

# Clean target (optional, customize as needed)
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__

# Phony targets
.PHONY: all run clean