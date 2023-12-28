import csv

# import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd


# Fonction pour lire le contenu d'un fichier
def lire_contenu_fichier(nom_fichier):
    with open(nom_fichier, "r", encoding="utf-8") as file:
        contenu = file.read()
    return contenu


# Fonction pour diviser le contenu en chunks (phrases)
def diviser_en_chunks(texte):
    phrases = sent_tokenize(texte)
    return phrases


# Fonction pour tokeniser une phrase en mots et créer un array de 0 avec gestion spécifique des tokens
def tokeniser_phrase(phrase):
    mots = word_tokenize(phrase)
    zeros = []
    is_per_tag = False
    filtered_words = []

    for mot in mots:
        if mot == "<":
            is_per_tag = True
            continue
        elif mot == "PER":
            continue
        elif mot == "/PER":
            is_per_tag = False
            continue
        elif mot == ">":
            continue

        if is_per_tag:
            zeros.append(
                2
            )  # Assigner 2 aux mots entre les balises <PER> et </PER>
            filtered_words.append(mot)
        else:
            zeros.append(0)
            filtered_words.append(mot)

    return filtered_words, zeros


# Fonction pour écrire dans un fichier CSV
def ecrire_csv(nom_fichier, data):
    with open(nom_fichier, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "tokens", "ner_tags"])
        for idx, (mots, zeros) in enumerate(data):
            writer.writerow([idx, mots, zeros])


if __name__ == "__main__":
    chemin_fichier = "../data/test_set/prelude_a_fondation/chapter_1.labeled"
    contenu = lire_contenu_fichier(chemin_fichier)
    phrases = diviser_en_chunks(contenu)
    donnees = [tokeniser_phrase(phrase) for phrase in phrases]
    nom_fichier_csv = "../data/finetuning_data/chapter_1.csv"
    ecrire_csv(nom_fichier_csv, donnees)

    chemin_fichier = "../data/test_set/prelude_a_fondation/chapter_2.labeled"
    contenu = lire_contenu_fichier(chemin_fichier)
    phrases = diviser_en_chunks(contenu)
    donnees = [tokeniser_phrase(phrase) for phrase in phrases]
    nom_fichier_csv = "../data/finetuning_data/chapter_2.csv"
    ecrire_csv(nom_fichier_csv, donnees)

    df = pd.read_csv("../data/finetuning_data/chapter_1.csv")
    df.to_parquet("../data/finetuning_data/chapter_1.parquet")

    df = pd.read_csv("../data/finetuning_data/chapter_2.csv")
    df.to_parquet("../data/finetuning_data/chapter_2.parquet")
