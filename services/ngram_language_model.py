import re
from collections import Counter, defaultdict
from typing import List, Dict, Union, Tuple

JOINING_STR: str = " "
REGEX_ENDING_PUNCTUATION_MARKS = r"([.!?\n]\s*)"


class NGramModel:
    STARTING_TOKEN: str = "<TEXT>"
    ENDING_TOKEN: str = "</TEXT>"
    SPLITING_TOKENIZER: str = " "
    NGRAN_MINIMUM_OCCURRENCE_THRESHOLD: int = 6
    MAX_LENGTH_GENERATED_TOKENS: int = 50

    def __init__(
        self,
        texts: List[str],
        max_ngram_size: int,
        minimum_occurrence_threshold: int = NGRAN_MINIMUM_OCCURRENCE_THRESHOLD,
    ):
        """Initializes the NGramModel with a corpus of text, n-gram size, and a minimum occurrence threshold."""
        self.texts = texts
        self.corpus = "".join(self.texts)
        self.max_ngram_size = max_ngram_size
        self.minimum_occurrence_threshold = minimum_occurrence_threshold
        self.tokenized_corpus = self.tokenize(self.tag_and_wrap_sentences(self.corpus))
        self.tokens = list(set(self.tokenized_corpus))
        self.ngrams = defaultdict(Counter)
        self.conditional_probabilities = defaultdict(lambda: defaultdict(float))

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenizes the text into words."""
        return re.findall(r"\w+|[.,!?;]", text)  # Improved tokenization

    @classmethod
    def tag_and_wrap_sentences(cls, text: str) -> str:
        """Wraps sentences with start/end tokens."""
        sentences = re.split(REGEX_ENDING_PUNCTUATION_MARKS, text)
        sentences = [
            sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)
        ]
        return JOINING_STR.join(
            [
                f"{cls.STARTING_TOKEN} {sentence.strip()} {cls.ENDING_TOKEN}"
                for sentence in sentences
            ]
        )

    def extract_ngrams(self, tokenized_corpus: List[str], n: int) -> List[Tuple[str]]:
        """Extracts n-grams from the tokenized corpus."""
        return [
            tuple(tokenized_corpus[i : i + n])
            for i in range(len(tokenized_corpus) - n + 1)
        ]

    def calculate_conditional_probabilities(
        self, ngram_counts: Counter, n_minus_1_counts: Counter
    ) -> Dict[Tuple[str], Dict[str, float]]:
        """Calculates the conditional probabilities of n-grams given (n-1)-grams."""
        probabilities = defaultdict(lambda: defaultdict(float))
        for ngram, ngram_count in ngram_counts.items():
            if ngram_count >= self.minimum_occurrence_threshold:
                context = ngram[:-1]  # The (n-1)-gram (everything except the last word)
                word = ngram[-1]  # The final word in the n-gram
                probabilities[context][word] = ngram_count / n_minus_1_counts[context]
        return probabilities

    def find_max_probabilities(
        self, probabilities: Dict[Tuple[str], Dict[str, float]]
    ) -> Dict[Tuple[str], str]:
        """Finds the word with the highest conditional probability for each (n-1)-gram context."""
        return {
            context: max(words, key=words.get)
            for context, words in probabilities.items()
        }

    def fit(self):
        """Fits the model by computing n-grams and conditional probabilities."""
        for ngram_size in range(1, self.max_ngram_size + 1):
            ngram = self.extract_ngrams(self.tokenized_corpus, ngram_size)
            self.ngrams[ngram_size] = Counter(ngram)

        for ngram_size in range(1, self.max_ngram_size):
            conditional_probabilities = self.calculate_conditional_probabilities(
                self.ngrams[ngram_size + 1], self.ngrams[ngram_size]
            )
            self.conditional_probabilities[ngram_size + 1] = (
                self.find_max_probabilities(conditional_probabilities)
            )

    def predict_next_token(self, tokens: tuple) -> Union[str, None]:
        """Predicts the next token given the current sequence of tokens."""
        for ngram_size in range(self.max_ngram_size, 1, -1):
            context = tokens[-(ngram_size - 1) :]
            next_token = self.conditional_probabilities.get(ngram_size, {}).get(
                context, None
            )
            if next_token:
                return next_token
        return None

    def predict(
        self, text: str, max_length_generated_tokens: int = MAX_LENGTH_GENERATED_TOKENS
    ) -> str:
        """Generates text by predicting the next tokens based on the input text."""
        tokens = self.tokenize(text)
        generated_tokens = tokens[:]  # Start with the original tokens

        ngram_tokens = tuple(tokens[-k] for k in range(self.max_ngram_size - 1, 0, -1))
        for _ in range(max_length_generated_tokens):
            new_token = self.predict_next_token(ngram_tokens)
            if new_token is None or new_token == self.ENDING_TOKEN:
                break
            ngram_tokens = ngram_tokens[1:] + (new_token,)
            generated_tokens.append(new_token)

        return JOINING_STR.join(generated_tokens)
