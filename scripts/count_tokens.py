from tqdm import tqdm
from vroom.GPTTokenizer import GPTTokenizer
from vroom.NER import read_file


def count_tokens():
    """
    Generates a submission file from the texts in the data/kaggle directory.
    """

    books = [
        (list(range(1, 20)), "paf"),
        (list(range(1, 19)), "lca"),
    ]

    total = 0
    for chapters, book_code in tqdm(books):
        for chapter in tqdm(chapters):
            if book_code == "paf":
                path = f"data/kaggle/prelude_a_fondation/chapter_{chapter}.txt.preprocessed"
            else:
                path = f"data/kaggle/les_cavernes_d_acier/chapter_{chapter}.txt.preprocessed"

            content = read_file(path)
            tokenizer = GPTTokenizer()
            tokenizer.tokenize(content)
            count_tokens = tokenizer.count_tokens()
            print("path : ", path)
            print("count_tokens : ", count_tokens)
            total += count_tokens
    print("total : ", total)


count_tokens()
