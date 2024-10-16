"""Code for calling the generating a text."""

from sys import argv

from TextGeneration.utils.files import json_to_schema, schema_to_json
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import InputSchema, OutputSchema

from utils import load_ngram_model


def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """
    # Reading input data
    input_schema = json_to_schema(file_str_path=file_str_path, input_schema=InputSchema)

    # Predicting texts
    generated_texts = []
    ngram_model = load_ngram_model(input_schema.trained_model)
    for input_text in input_schema.texts:
        text = Preprocessor.clean(text=input_text)
        generated_texts.append(ngram_model.predict(text))

    # Printing generated texts
    output_schema = OutputSchema(generated_texts=generated_texts)
    schema_to_json(file_path=input_schema.output_file, schema=output_schema)


if __name__ == "__main__":
    main_generate(file_str_path=argv[1])
