r"""  Package for tokenization tasks.

Authors
--------
 * Gabriel DESBOUIS 2023
"""

import tiktoken


class GPTTokenizer:
    """
    A wrapper for the GPT tokenizer.
    It allows to tokenize a text and count the number of tokens.
    """
    def __init__(self, model: str = "gpt-4") -> None:
        self.encoding = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenizes the given text.
        Args:
            text (str): The text to tokenize.
        Returns:
            list[int]: The list of tokens.
        """
        self.tokenized: bool = True
        self.tokens = self.encoding.encode(text)
        return self.tokens

    def count_tokens(self) -> int:
        """
        Counts the number of tokens in the text.
        Returns:
            int: The number of tokens in the text.
        """
        if not self.tokenized:
            raise Exception(
                "You must tokenize the text before counting the tokens."
            )
        return len(self.tokens)
