import pickle
from services.ngram_language_model import NGramModel


def save_ngram_model(model: NGramModel, filename: str):
    """Saves the NGram model to a file."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_ngram_model(filename: str) -> NGramModel:
    """Loads the NGram model from a file."""
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model
