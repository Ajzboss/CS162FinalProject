#!/bin/bash
python3 -m venv roberta_env
source roberta_env/bin/activate
pip install --upgrade pip
pip install transformers datasets scikit-learn pandas jupyter 'accelerate>=0.26.0'
echo "Virtual environment 'roberta_env' set up. Run 'source roberta_env/bin/activate' before launching Jupyter."
