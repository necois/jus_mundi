"""Code for calling the training of the model."""

from sys import argv
from typing import List
from pathlib import Path, PosixPath

from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema

from services.ngram_language_model import NGramModel
from utils import save_ngram_model


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_schema = json_to_schema(
        file_str_path=file_str_path, input_schema=TrainingInputSchema
    )

    # Loading raw documents
    training_texts: List[str] = []
    for training_line in read_dir(dir_path=training_schema.input_folder):
        training_texts.append(Preprocessor.clean(text=training_line))

    # Training NGramModel
    ngram_model = NGramModel(training_texts, training_schema.max_n_gram)
    ngram_model.fit()
    assert isinstance(ngram_model.predict("This is the story of", 20), str)
    save_ngram_model(ngram_model, Path.cwd() / training_schema.trained_model)


if __name__ == "__main__":
    training_data_path: PosixPath = Path.cwd() / argv[1]
    main_train(file_str_path=training_data_path)
