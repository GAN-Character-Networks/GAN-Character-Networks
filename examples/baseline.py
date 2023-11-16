from vroom.baseline import get_cooccurences_from_text

def baseline(text):
    return get_cooccurences_from_text(text)

if __name__ == "__main__":
    path = "data/kaggle/les_cavernes_d_acier/chapter_1.txt.preprocessed"

    print(baseline(path))