from services.ngram_language_model import NGramModel
import dill as pickle


def save_ngram_model(ngram_model: NGramModel, path: str) -> None:
    """Saves an NGramModel to the specified path using dill."""
    with open(path, "wb") as f:
        pickle.dump(ngram_model, f)


def load_ngram_model(path: str) -> NGramModel:
    """Loads an NGramModel from the specified path using dill."""
    with open(path, "rb") as f:
        ngram_model = pickle.load(f)
    return ngram_model
