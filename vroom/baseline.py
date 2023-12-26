r"""  Pipeline for cooccurences extraction with baseline methods

Authors
--------
 * Nicolas Bataille 2023
 * Adel Moumen 2023
 * Gabriel Desbouis 2023
"""
from vroom.NER import get_entities_from_file
from vroom.alias import get_aliases_fuzzy_partial_token
from vroom.cooccurences import get_cooccurences
from vroom.loggers import JSONLogger
from openai import OpenAI
import json


def get_cooccurences_from_text(path: str):
    """
    Get the coocurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """

    entities, chunks = get_entities_from_file(path)
    return get_cooccurences(chunks, entities)

def find_cooccurrences_aliases(cooccurrences, aliases):
    no_alias_1_list = []
    no_alias_2_list = []
    cooccurrences_aliases = []

    for cooc in cooccurrences:
        no_alias_1 = True
        no_alias_2 = True

        for alias in aliases:
            if cooc[0] in alias:
                no_alias_1 = False
            if cooc[1] in alias:
                no_alias_2 = False

        if no_alias_1:
            no_alias_1_list.append(cooc[0])
        if no_alias_2:
            no_alias_2_list.append(cooc[1])

    print("no alias 1 list: ", no_alias_1_list)
    print("no alias 2 list: ", no_alias_2_list)

    for cooccurence in cooccurrences:
        cooc_1_aliases = [
            alias
            for alias in aliases
            if cooccurence[0].lower() in [a.lower() for a in alias]
        ]
        cooc_2_aliases = [
            alias
            for alias in aliases
            if cooccurence[1].lower() in [a.lower() for a in alias]
        ]

        if cooc_1_aliases and cooc_2_aliases and cooc_1_aliases[0] != cooc_2_aliases[0]:
            cooccurrences_aliases.append((cooc_1_aliases[0], cooc_2_aliases[0]))

    print("aliases: ", aliases)
    return cooccurrences_aliases


def get_cooccurences_with_aliases(path: str, logger: JSONLogger = None):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.
        logger (JSONLogger, optional): The logger to save the aliases. Defaults to None.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    entities, chunks = get_entities_from_file(path)
    cooccurences = get_cooccurences(chunks, entities)
    entities_unfold = [entity for sublist in entities for entity in sublist]
    aliases = get_aliases_fuzzy_partial_token(entities_unfold, 99)

    if logger is not None: 
        saves = {}
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(set([entity["word"] for entity in entities_unfold]))
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        logger(saves)

    cooccurences_aliases = find_cooccurrences_aliases(cooccurences, aliases)
    return cooccurences_aliases


# TODO: Idée de solving d'alias : Récup tout les alias de tout les chapitres du livre, puis construire par chapitre la liste d'alias correspondante


def get_cooccurences_with_aliases_and_gpt(path: str):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    system_prompt = r"""
Tu es un expert dans la résolution d'alias de personnages de fiction.
Ton but est d'à partir d'une liste de personnages de déterminer qui est qui en faisant un clustering.
En effet, ces noms représentent des personnages issus du livre 'Fondation' d'Isaac Asimov. Cependant,
certains personnages ont plusieurs noms, et tu cherches a trouver qui est la bonne personne. Chaque cluster représente donc un personnage avec tous ses alias.
Tu dois faire attention à la sémantique des mots, notamment le genre, les  titres, etc.

Donne ta réponse sous le format JSON suivant, et ne dévie pas de cette tâche :

{
   '0' : [nom_de_alias_1, ...],
}

Chaque entrée du JSON correspond à un personnage et à l'ensemble de ses alias. La clé est un chiffre qui représente uniquement sa position dans le JSON. La liste associée correspond à l'ensemble des alias de ce personnage.

Example :

Personnages :
Cléon
CLÉON Ier-
Empereur
Demerzel
Hari Seldon
Empereurs

Sorti JSON attendu :
{
    '0' : ['Cléon', 'CLÉON Ier-', 'Empereur', 'Empereurs'],
    '1' : ['Demerzel'],
    '2' : ['Hari Seldon'],
}


Fait le pour les personnages suivant et essaye de trouver le plus d'alias possible :
    """

    entities, chunks = get_entities_from_file(path, device="cuda")

    entities = [entity for sublist in entities for entity in sublist]
    word_entities = [entity["word"] for entity in entities]
    print("entities = ", set(word_entities))
    user_prompt = "\n".join(set(word_entities))

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # gpt-3.5-turbo-1106, gpt-4-1106-preview
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=42,
        temperature=0,
        response_format={"type": "json_object"},
    )

    generated_content = response.choices[0].message.content
    generated_content = json.loads(generated_content)

    aliases = []
    for key in generated_content:
        aliases.append(generated_content[key])

    print("aliases : ", aliases)

    cooccurences = get_cooccurences(chunks, entities)

    for cooc in cooccurences:
        no_alias_1 = True
        no_alias_2 = True
        for alias in aliases:
            if cooc[0] in alias:
                no_alias_1 = False
            if cooc[1] in alias:
                no_alias_2 = False
        if no_alias_1:
            print("no alias 1 : ", cooc[0])
        if no_alias_2:
            print("no alias 2 : ", cooc[1])

    cooccurences_aliases = []
    for cooccurence in cooccurences:
        cooc_1_aliases = [
            alias
            for alias in aliases
            if cooccurence[0].lower() in [a.lower() for a in alias]
        ][0]
        cooc_2_aliases = [
            alias
            for alias in aliases
            if cooccurence[1].lower() in [a.lower() for a in alias]
        ][0]
        cooccurences_aliases.append((cooc_1_aliases, cooc_2_aliases))
    return cooccurences_aliases
