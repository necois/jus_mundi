# Jus Mundi ML Assessment

This project involves training and generating text using custom n-gram models. It provides a configurable setup for training models and generating text, along with unit tests to ensure code functionality.

### Setup Instructions
- Create a virtual environment and activate it:
- Upgrade pip: `pip install --upgrade pip`
- Install the required dependencies: `pip install -r requirements.txt`

## Run local test to ensure compliance with exercise instructions
`python -m unittest tests/test_ngram_models.py`

# Train NGramModel
`python -m TextGeneration.train data/config_jsons/training.json`

# Generate some predictions
`python -m TextGeneration.generate data/config_jsons/input.json`
