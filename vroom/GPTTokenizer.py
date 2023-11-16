import tiktoken


class GPTTokenizer:
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str):
        self.tokenized = True
        self.tokens = self.encoding.encode(text)
        return self.tokens

    def count_tokens(self):
        if not self.tokenized:
            raise Exception(
                "You must tokenize the text before counting the tokens."
            )
        return len(self.tokens)
