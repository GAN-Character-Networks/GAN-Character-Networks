import os

import vroom.GPTTokenizer

cavernes_files = os.listdir("data/kaggle/les_cavernes_d_acier")
prelude_files = os.listdir("data/kaggle/prelude_a_fondation")

total_tokens = 0

tokenizer = vroom.GPTTokenizer.Tokenizer()

for file in cavernes_files:
    with open("data/kaggle/les_cavernes_d_acier/" + file, "r") as f:
        text = f.read()
        tokenizer.tokenize(text)
        total_tokens += tokenizer.count_tokens()

print("Nombre de token dans le corpus des Cavernes d'acier :")
print(total_tokens)

for file in prelude_files:
    with open("data/kaggle/prelude_a_fondation/" + file, "r") as f:
        text = f.read()
        tokenizer.tokenize(text)
        total_tokens += tokenizer.count_tokens()


print(total_tokens)
