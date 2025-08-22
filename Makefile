# Minimal Makefile: create .venv, install, run environmental scripts

VENV := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
REQ   := requirements.txt

.PHONY: all install_requirements environmental clean

all: install_requirements environmental

# Create .venv (once) and upgrade pip
$(PY):
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install_requirements: $(PY)
	$(PIP) install -r $(REQ)

environmental: $(PY)
	$(PY) src/advml_hackaton/environmental/data_preprocessing.py
	$(PY) src/advml_hackaton/environmental/hyperparameter_optim.py
	$(PY) src/advml_hackaton/environmental/select_best_model.py
	$(PY) src/advml_hackaton/environmental/train.py

monitor_ble:
	$(PY) src/advml_hackaton/environmental/monitor_ble.py

clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__