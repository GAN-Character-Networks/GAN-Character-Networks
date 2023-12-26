from openai import OpenAI
from vroom.NER import *
import os
from vroom.GPTTokenizer import GPTTokenizer
import json
from vroom.GraphManager import GraphManager
from vroom.cooccurences import get_cooccurences
import nltk
import html
import pandas as pd
from tqdm import tqdm

OPENAI_API_KEY="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"
total_tokens_sent = 0
total_tokens_received = 0
client = OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-3.5-turbo-1106" # gpt-4-1106-preview

tokenizer = GPTTokenizer(model)

def cooccurences_from_gpt_json(text, entities):
    # Tokeniser le texte
    tokens = nltk.word_tokenize(text)
    # Créer un dictionnaire pour stocker les positions de chaque alias
    alias_positions = {}

    # Parcourir les entités pour trouver les positions de leurs alias
    for _, info in entities.items():
        aliases = info['aliases']
        for alias in aliases:
            alias_words = nltk.word_tokenize(alias)
            alias_length = len(alias_words)

            # Parcourir les tokens pour trouver les occurrences de l'alias
            for i in range(len(tokens) - alias_length + 1):
                if tokens[i:i + alias_length] == alias_words:
                    # Ajouter la position de début de l'alias dans le texte
                    alias_positions.setdefault(alias, []).append(i)

    cooccurrences = []
    for entity1, aliases1 in entities.items():
        for entity2, aliases2 in entities.items():
            if entity1 != entity2:
                # Vérifiez chaque combinaison d'alias pour la cooccurrence
                for alias1 in aliases1['aliases'] + [entity1]:
                    for alias2 in aliases2['aliases'] + [entity2]:
                        if alias1 in alias_positions and alias2 in alias_positions:
                            if any(abs(pos1 - pos2) <= 25 for pos1 in alias_positions[alias1] for pos2 in alias_positions[alias2]):
                                # Ajouter la paire de listes d'alias à la liste de cooccurrences
                                cooccurrences.append((aliases1['aliases'], aliases2['aliases']))
                                break  # Arrêtez de chercher d'autres occurrences pour cette paire
    return cooccurrences

def build_entities_with_gpt(text, len_chunk):
    # Appel à l'API de GPT-3 pour générer les alias
    system_prompt = """
    C'est un travail capital que je te demande de faire, c'est un enjeu énorme pour ma carrière. Si tu fais un excellent travail, je te donnerai un pourboire.
    Tu es un expert dans l'étude des personnages dans les romans. J'ai un texte annoté par un modèle d'NER, et je n'ai gardé que les entités qu'il a estimé être des personnages. Elles sont situées entre les balises <PER></PER>. J'ai besoin qu'à partir de ce texte, tu me construises un JSON qui contient tous les personnages et leurs alias associé. Il se peut que certaines entités soit marquées comme des personnages mais ne le soient pas vraiment, elles peuvent faire référence à des lieux ou des expressions par exemple, tu ne dois pas les ajouter au JSON. Tu dois le faire comme dans cet exemple : 

    Input : 
    <start>
    Mathématicien  <PER> CLÉON Ier </PER>  — ... dernier  <PER> Empereur </PER>  galactique de la dynastie Entun. Né en l'an 11988 de l'Ère Galactique, la même année que  <PER> Hari  Seldon  </PER> . (On pense que la date de naissance de  <PER> Seldon </PER> , que certains estiment douteuse, aurait pu être « ajustée » pour coïncider avec celle de  <PER> Cléon </PER>  que  <PER> Seldon </PER> avait trouvé sur <PER> Trantor </PER> en 12020.)
    <end>

    Output : 
    <start>
    {
    " CLÉON Ier ":{
        "aliases":[
            "Cléon Ier",
            "Cléon",
            "Empereur"
        ]
    },
    "Hari Seldon":{
        "aliases":[
            "Hari Seldon",
            "Seldon"
        ]
    }
    }
    <end>

    Pour chaque entrée dans le JSON, tu devras extraire un seul personnage du texte, et y associer tout ses alias. Rajoute la clé de chaque personnage dans ses alias.

    Voici l'input :
    """
    len_chunk = 15000
    chunks = chunk_text(text, len_chunk)

    json_entities = []
    len_prompt = len(tokenizer.tokenize(system_prompt))

    for chunk in chunks:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
            seed=42,
            temperature=0,
            response_format={ "type": "json_object" },
            )

        json_entities.append(response.choices[0].message.content)
        tokens_sent = len(tokenizer.tokenize(chunk))+len_prompt
        print("Tokens sent : ", tokens_sent)
        tokens_received = len(tokenizer.tokenize(response.choices[0].message.content))
        print("Tokens received : ", tokens_received)
        total_tokens_sent += tokens_sent
        total_tokens_received += tokens_received

    return json_entities

def compile_json(json_entities):
    """
    json_entities : liste de json
    """
    # Dictionnaire global pour le résultat fusionné
    merged_entities = {}

    # Vérifier s'il n'y a pas plus de deux sauts de ligne consécutifs ((\n){2,}) si oui, les remplacer par un seul
    json_entities = [re.sub(r"((\n){2,})", "\n", json_entity) for json_entity in json_entities]
    for entity_str in json_entities:
        # Convertir la chaîne JSON en dictionnaire
        try:
            entity_dict = json.loads(entity_str)
        except:
            print("entity_str mal formaté : ", entity_str)
            continue
        for key, value in entity_dict.items():
            if key not in merged_entities:
                # Si la clé n'existe pas, l'ajouter au dictionnaire global
                merged_entities[key] = value
            else:
                # Si la clé existe déjà, fusionner ou remplacer selon vos besoins
                # Par exemple, fusionner les listes d'alias
                existing_aliases = set(merged_entities[key]['aliases'])
                new_aliases = set(value['aliases'])
                merged_aliases = list(existing_aliases.union(new_aliases))
                merged_entities[key]['aliases'] = merged_aliases

    # Convertir le dictionnaire global en chaîne JSON
    return json.dumps(merged_entities, indent=4, ensure_ascii=False)

def filter_entities(all_entities):
    filtering_prompt = """
    C'est un travail capital que je te demande de faire, c'est un enjeu énorme pour ma carrière. Si tu fais un excellent travail, je te donnerai un pourboire.
    Tu es un expert dans l'étude des personnages dans les romans. J'ai un texte annoté par un modèle d'NER, et je n'ai gardé que les entités qu'il a estimé être des personnages.
    Nous avons eu ce JSON comme résultat mais il se peut qu'il y ait des doublons. Tu dois donc les supprimer. Si tu vois qu'un personnage a un alias qui est aussi un personnage, privilégie l'alias au personnage. Tu as stricte interdiction d'inventer des alias ou des personnages.

    Voici un exemple :
    Voici l'input :
    <start>
    {
        " CLÉON Ier ":{
            "aliases":[
                "Cléon Ier",
                "Cléon",
                "Empereur",
                "CLÉON Ier"
            ]
        },
        "Hari Seldon":{
            "aliases":[
                "Hari Seldon",
                "Seldon"
            ]
        }
        "Empereur":{
            "aliases":[
                "Empereur"
            ]
        }
        "Seldon":{
            "aliases":[
                "Seldon"
            ]
        }
    }
    <end>

    Output :
    <start>
    {
        " CLÉON Ier ":{
            "aliases":[
                "Cléon Ier",
                "Cléon",
                "Empereur"
            ]
        },
        "Hari Seldon":{
            "aliases":[
                "Hari Seldon",
                "Seldon"
            ]
        }
    }
    <end>
    """
    len_prompt = len(tokenizer.tokenize(filtering_prompt))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": filtering_prompt},
            {"role": "user", "content": json.dumps(all_entities, indent=4, ensure_ascii=False)},
        ],
        seed=42,
        temperature=0,
        response_format={ "type": "json_object" },
        )

    tokens_sent = len(tokenizer.tokenize(json.dumps(all_entities, indent=4, ensure_ascii=False)))+len_prompt
    print("Tokens sent : ", tokens_sent)
    tokens_received = len(tokenizer.tokenize(response.choices[0].message.content))
    print("Tokens received : ", tokens_received)
    total_tokens_sent += tokens_sent
    total_tokens_received += tokens_received
    return json.loads(response.choices[0].message.content)


def generate_submission():
    """
    Generates a submission file from the texts in the data/kaggle directory.
    """

    books = [
        (list(range(1, 20)), "paf"),
        (list(range(1, 19)), "lca"),
    ]

    df_dict = {"ID": [], "graphml": []}

    for chapters, book_code in tqdm(books):
        for chapter in tqdm(chapters):
            if book_code == "paf":
                path = f"data/kaggle/prelude_a_fondation/chapter_{chapter}.txt.preprocessed"
            else:
                path = f"data/kaggle/les_cavernes_d_acier/chapter_{chapter}.txt.preprocessed"
            print("Processing : ", path)
            graph_manager = GraphManager()

            # Specifique à ce fichier
            with open(path, "r") as f:
                text = f.read()
            entities = build_entities_with_gpt(text, 5000)
            entities = compile_json(entities)
            entities = filter_entities(entities)
            coocurrences = cooccurences_from_gpt_json(text, entities)
            graph_manager.add_cooccurrences(coocurrences)
            df_dict["ID"].append(f"{book_code}{chapter-1}")
            df_dict["graphml"].append("".join(html.unescape(s) if isinstance(s, str) else s for s in graph_manager.generate_graph()))

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("submission.csv")




if __name__ == "__main__":
    generate_submission()
    print("Prix de la submission : ", total_tokens_sent/1000*0.001 + total_tokens_received/1000*0.002, " $")