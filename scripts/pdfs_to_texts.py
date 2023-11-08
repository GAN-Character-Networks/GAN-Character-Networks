"""
"""
import os

if __name__ == "__main__":
    for filename in os.listdir("data/pdfs"):
        if filename.endswith(".pdf"):
            # Ajouter des test et des try/except
            try:
                os.system(
                    "pdftotext -x 100 -y 1000 -W 10000 -H 10000 -r 1000 data/pdfs/"
                    + filename
                    + " data/txts/"
                    + filename[:-4]
                    + ".txt"
                )
            except Exception as e:
                print(
                    f"An error occurred during the conversion of the pdfs : {str(e)}"
                )
                continue
        else:
            continue

    print("The pdfs have been converted.")
