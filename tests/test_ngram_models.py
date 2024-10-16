import unittest
import numpy as np

from services.ngram_language_model import NGramModel


class TestNGramModel(unittest.TestCase):

    def setUp(self):
        """Set up test data for the NGramModel class."""
        # Use the provided training text
        self.texts = [
            "Jus Mundi is the leader in Global Legal Intelligence. And now, with Jus AI, we provide an AI-Powered Global Legal Intelligence too. Jus Mundi is based in Paris.",
        ]
        # Initialize NGram model with n = 2 for bigrams
        self.model = NGramModel(
            self.texts, max_ngram_size=2, minimum_occurrence_threshold=1
        )

    def test_bigram_probabilities(self):
        """Test that the bigram probabilities are calculated correctly."""
        # Fit the model
        self.model.fit()

        # The expected bigram counts based on the provided text:
        expected_bigrams = {
            ("<TEXT>", "Jus"): 1,
            ("Jus", "Mundi"): 2,
            ("Jus", "AI,"): 1,
            ("in", "Global"): 1,
            ("in", "Paris."): 1,
            ("Mundi", "is"): 2,
            ("Paris.", "</TEXT>"): 1,
        }

        # Validate if the bigram counts match
        for bigram, count in expected_bigrams.items():
            self.assertEqual(
                self.model.ngrams[2][bigram],
                count,
                f"Expected count for {bigram} to be {count}",
            )

        # Check calculated probabilities based on the bigram counts
        # P(Jus | <TEXT>) = C(<TEXT> Jus) / C(<TEXT>) = 1 / 1 = 1.00
        prob_Jus_given_text = self.model.ngram_conditional_probabilities_dict[2][
            ("<TEXT>",)
        ]["Jus"]
        self.assertEqual(
            prob_Jus_given_text,
            1.0,  # Expected probability of 'Jus' given '<TEXT>' to be 1.00
            "Expected probability of 'Jus' given '<TEXT>' to be 1.00",
        )

        # P(Mundi | Jus) = C(Jus Mundi) / C(Jus) = 2 / 3 = 0.67
        prob_Mundi_given_Jus = self.model.ngram_conditional_probabilities_dict[2][
            ("Jus",)
        ]["Mundi"]

        self.assertEqual(
            np.round(prob_Mundi_given_Jus, 2),
            0.67,  # Expected probability of 'Mundi' given 'Jus' to be approximately 0.66
            msg="Expected probability of 'Mundi' given 'Jus' to be 0.67",
        )

        # P(AI | Jus) = C(Jus AI) / C(Jus) = 1 / 3 = 0.33
        prob_AI_given_Jus = self.model.ngram_conditional_probabilities_dict[2][
            ("Jus",)
        ]["AI,"]
        self.assertEqual(
            np.round(prob_AI_given_Jus, 2),
            0.33,
            "Expected probability of 'AI,' given 'Jus' to be 0.33",
        )

        # P(Global | in) = C(in Global) / C(in) = 1 / 2 = 0.50
        prob_Global_given_in = self.model.ngram_conditional_probabilities_dict[2][
            ("in",)
        ]["Global"]
        self.assertEqual(
            prob_Global_given_in,
            0.5,
            "Expected probability of 'Global' given 'in' to be 0.50",
        )

        # P(Paris. | in) = C(in Paris.) / C(in) = 1 / 2 = 0.50
        prob_Paris_given_in = self.model.ngram_conditional_probabilities_dict[2][
            ("in",)
        ]["Paris."]
        self.assertEqual(
            prob_Paris_given_in,
            0.5,
            "Expected probability of 'Paris.' given 'in' to be 0.50",
        )

        # P(is | Mundi) = C(Mundi is) / C(Mundi) = 2 / 2 = 1.00
        prob_is_given_Mundi = self.model.ngram_conditional_probabilities_dict[2][
            ("Mundi",)
        ]["is"]
        self.assertEqual(
            prob_is_given_Mundi,
            1.0,
            "Expected probability of 'is' given 'Mundi' to be 1.00",
        )

        # P(</TEXT> | Paris.) = C(Paris. </TEXT>) / C(</TEXT>) = 1 / 1 = 1.00
        prob_end_given_Paris = self.model.ngram_conditional_probabilities_dict[2][
            ("Paris.",)
        ]["</TEXT>"]
        self.assertEqual(
            prob_end_given_Paris,
            1.0,
            "Expected probability of '</TEXT>' given 'Paris.' to be 1.00",
        )

    def test_text_generation(self):
        """Test text generation using the trained model."""
        # Fit the model first
        self.model.fit()

        # Provide a starting sentence for text generation
        starting_text = "<TEXT> Jus Mundi"
        generated_text = self.model.predict(
            starting_text, max_length_generated_tokens=10
        )

        # Check if the generated text starts with the expected tokens
        self.assertTrue(
            generated_text.startswith(starting_text),
            f"Expected generated text to start with '{starting_text}'",
        )

        # We can also validate if the generated tokens align with bigram probabilities
        # In this case, the next tokens after 'Jus Mundi' should be 'is' or 'AI'
        following_token = generated_text.split(" ")[len(starting_text.split(" "))]
        self.assertIn(
            following_token,
            ["is", "AI"],
            f"Expected generated token to be 'is' or 'AI' but got {following_token}",
        )


if __name__ == "__main__":
    unittest.main()
