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
from vroom.NER import *
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

def find_cooccurences_aliases(cooccurences, aliases):
    no_alias_1_list = []
    no_alias_2_list = []
    cooccurrences_aliases = []

    for cooc in cooccurences:
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

    for cooccurence in cooccurences:
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



    cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
    print("cooccurences_aliases: ", cooccurences_aliases)
    exit()
    return cooccurences_aliases


# TODO: Idée de solving d'alias : Récup tout les alias de tout les chapitres du livre, puis construire par chapitre la liste d'alias correspondante


def get_cooccurences_with_aliases_and_gpt(path: str, logger: JSONLogger = None):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    system_prompt = r"""
Tu es un expert dans les livres "La Fondation" de Isaac Asimov.
Ton but est d'à partir d'une liste de personnes de déterminer qui est qui en faisant un regroupement. Chaque regroupement représente une personne avec toutes ses références.

Pour réaliser ce regroupement, base toi sur la sémantique des mots, notamment le genre, les titres, les ressemblances typographiques, etc. 

Donne ta réponse sous le format JSON suivant, et ne dévie pas de cette tâche :

{
   '0' : [reference_1, …],
   '1' : [reference_1, …],
}

Chaque entrée du JSON correspond à un personnage et à l'ensemble de ses références. La clef est un chiffre qui donne la position dans le JSON. La position n’a pas d’importance.

Voici des exemples pour t'aider :

Example 1 : 

Input:
CLÉON Ier
Empereur
Cléon
Sire
l'Empereur
l'empereur Cléon
Hari Seldon
Seldon
Eto Demerzel
Demerzel
Lieutenant Alban Wellis
Wellis
Hummin
Trantor

Output:
{
    '0' : ['CLÉON Ier', 'Empereur', 'Cléon', 'Sire', 'l'Empereur', 'l'empereur Cléon'],
    '1' : ['Hari Seldon', 'Seldon', 'Eto Demerzel', 'Demerzel'],
    '2' : ['Lieutenant Alban Wellis', 'Wellis'],
    '3' : ['Hummin'],
}

Tu vas desormais recevoir une liste de personnes en input, et tu dois les regrouper s'ils se réfèrent à la même personne. Tu n'as pas le droit d'inventer de nouveaux personnages et tu dois uniquement utiliser cette liste. Les noms ne doivent pas être modifiés, y compris les espaces, apostrophes, etc. Si tu ne respectes pas ces règles, tu seras fortement pénalisé. 
Tu as le droit de supprimer des personnes si tu penses qu'ils ne sont pas des personnages du livre. En effet, le système qui prédit un personnage peut se tromper, et tu dois le corriger.

Fais-le pour les personnes suivantes et essaie de trouver les meilleurs regroupements possibles. Je compte sur toi, merci !
    """

    entities, chunks = get_entities_from_file(path, device="cuda")
    cooccurences = get_cooccurences(chunks, entities)
    entities = [entity for sublist in entities for entity in sublist]
    entities_to_remove = [
        "Spacetown",
        "ah",
        "le Spacien",
        "cle",
        "Ciel",
        "Williamsburg",
        # "B... Baley",
        "Trantor",
        "Il",
        "Aurora",
        "Novigor",
        "Bande",
        "Deux",
        "Qu"
    ]
    entities = [entity for entity in entities if entity["word"] not in entities_to_remove]

    word_entities = [entity["word"] for entity in entities]
    print("entities = ", set(word_entities))
    user_prompt = """

    Input : 
    """
    user_prompt += "\n".join(set(word_entities)) + "\nOutput : "
    experiment_details = """
    gpt-4-1106-preview avec 1 exemples de donné en prompt.
    """ #GPT-3.5-turbo-1106

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-4-1106-preview", # gpt-4-1106-preview
    }
    response = client.chat.completions.create(
        **params,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    generated_content = response.choices[0].message.content
    generated_content = json.loads(generated_content)

    aliases = []
    for key in generated_content:
        aliases.append(generated_content[key])

    if logger is not None: 
        saves = {}
        saves["experiment_details"] = experiment_details
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(set(word_entities))
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        saves["gpt"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_content": generated_content,
            "params": params,
        }
        logger(saves)

    return find_cooccurences_aliases(cooccurences, aliases) 

def get_cooccurences_with_aliases_and_gpt_NER(path: str, logger: JSONLogger = None):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    system_prompt = """
    Je suis un excellent linguiste. La tâche consiste à étiqueter les entités de type "Personnages" dans la phrase donnée. Ces phrases sont issus des livres de science-fiction "Le cycle des Fondations" d'Isaac Asimov. Voici quelques exemples :

    Input : Dors tendit la main pour le prendre et pianota sur les touches. Il lui fallut un moment car la disposition n’était pas tout à fait orthodoxe, mais elle parvint à allumer l’écran et à inspecter les pages.

    Output : @@Dors## tendit la main pour le prendre et pianota sur les touches. Il lui fallut un moment car la disposition n’était pas tout à fait orthodoxe, mais elle parvint à allumer l’écran et à inspecter les pages.

    Input : C’est vraiment enfantin, dit Goutte-de-Pluie Quarante- cinq. Nous pouvons vous montrer. — Nous allons vous préparer un bon repas bien nourrissant », dit Goutte-de-Pluie Quarante-trois.

    Output : C’est vraiment enfantin, dit @@Goutte-de-Pluie Quarante- cinq##. Nous pouvons vous montrer. — Nous allons vous préparer un bon repas bien nourrissant », dit @@Goutte-de-Pluie Quarante-trois##.

    Input : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
    dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
    année que Hari Seldon. (On pense que la date de naissance de
    Seldon, que certains estiment douteuse, aurait pu être
    « ajustée » pour coïncider avec celle de Cléon que Seldon, peu
    après son arrivée sur Trantor, est censé avoir rencontré.)

    Output : Mathématicien @@CLÉON Ier##— ... dernier @@Empereur## galactique de la
    dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
    année que @@Hari Seldon##. (On pense que la date de naissance de
    @@Seldon##, que certains estiment douteuse, aurait pu être
    « ajustée » pour coïncider avec celle de @@Cléon## que @@Seldon##, peu
    après son arrivée sur Trantor, est censé avoir rencontré.)

    Input : Toutes les citations de l'Encyclopaedia Galactica reproduites ici
    proviennent de la 116e édition, publiée en 1020 E.F. par la Société
    d’édition de l'Encyclopaedia Galactica, Terminus, avec l'aimable
    autorisation des éditeurs.
    semblerait malgré tout qu’il puisse encore arriver des choses
    intéressantes. Du moins, à ce que j’ai entendu dire.
        — Par le ministre des Sciences ?
        — Effectivement. Il m’a appris que ce Hari Seldon a assisté
    à un congrès de mathématiciens ici même, à Trantor – ils
    l’organisent tous les dix ans, pour je ne sais quelle raison ; il
    aurait démontré qu’on peut prévoir mathématiquement

    Output : Toutes les citations de l'Encyclopaedia Galactica reproduites ici
    proviennent de la 116e édition, publiée en 1020 E.F. par la Société
    d’édition de l'Encyclopaedia Galactica, Terminus, avec l'aimable
    autorisation des éditeurs.
    semblerait malgré tout qu’il puisse encore arriver des choses
    intéressantes. Du moins, à ce que j’ai entendu dire.
        — Par le ministre des Sciences ?
        — Effectivement. Il m’a appris que ce @@Hari Seldon## a assisté
    à un congrès de mathématiciens ici même, à Trantor – ils
    l’organisent tous les dix ans, pour je ne sais quelle raison ; il
    aurait démontré qu’on peut prévoir mathématiquement

    Input : Seldon savait qu’il n’avait pas le choix, nonobstant les circonlocutions polies de l’autre, mais rien ne lui interdisait de chercher à en savoir plus : « Pour voir l’Empereur ?

    Output : @@Seldon## savait qu’il n’avait pas le choix, nonobstant les circonlocutions polies de l’autre, mais rien ne lui interdisait de chercher à en savoir plus : « Pour voir @@l’Empereur##, @@Sire## ?

    Je ne dévierais pas de cette tâche et ferais exactement comme dans les exemples. Je recopierais 
    le texte en Input et en Output j'ajouterais le texte et les balises. Si je ne trouve pas de Personnage,
    je ne mettrais pas de balise. Si je croise un personnage, avec de la ponctuation, je mettrais la balise
    avant la ponctuation sauf si c'est un mot composé avec des tirets. Garde les determinants avec le nom d'un personnage si tu peux.
    """
    from vroom.alias import get_aliases_fuzzy
    from vroom.cooccurences import get_cooccurences
    
    content = read_file(path)
    chunks = chunk_text_by_sentence(content, batch_size = 5) 
    
    entities = []
    experiment_details = """
    """ #GPT-3.5-turbo-1106

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-3.5-turbo-1106", # gpt-4-1106-preview
    }
    gpt_outputs = []
    import time
    for i, chunk in enumerate(chunks):
        print('*' * 50)
        print(chunk)
        user_prompt = f"""

        Input : {chunk}
        """ + "\n\n   Output : "
        
        response = client.chat.completions.create(
            **params,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        generated_content = response.choices[0].message.content
        print()
        print(generated_content)
        print()
        print('*' * 50)
        gpt_outputs.append(generated_content)

    if logger is not None: 
        saves = {}
        saves["experiment_details"] = experiment_details
        for i, (chunk, ner_chunk) in enumerate(zip(chunks, gpt_outputs)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "ner_chunk": ner_chunk,
            }
        saves["gpt"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_content": generated_content,
            "params": params,
        }
        logger(saves)

    tagged_file = tag_text_with_entities(path, gpt_entities)
    out_entities = get_positions_of_entities(tagged_file)
    cooccurences = get_cooccurences([tagged_file], [out_entities])

    print('entities = ', entities)
    entities = [{"word":entity} for entity in entities]
    aliases = get_aliases_fuzzy(entities, 99)
    print('alises = ', aliases)

    if logger is not None: 
        saves = {}
        saves["experiment_details"] = experiment_details
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(gpt_entities)
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        saves["gpt"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "generated_content": generated_content,
            "params": params,
        }
        logger(saves)
    

    return find_cooccurences_aliases(cooccurences, aliases)