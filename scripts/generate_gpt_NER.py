r""" Submission file for using GPT to generate the NER chunks + GPT self_verification + fuzzy partial token matching.

Authors
--------
 * Adel Moumen 2023, 2024
"""

from vroom.baseline import (
    get_aliases_fuzzy_partial_token,
    find_cooccurences_aliases,
    get_cooccurences,
)
from vroom.loggers import JSONLogger
from vroom.GraphManager import GraphManager
from openai import OpenAI
from tqdm import tqdm
import os
import pandas as pd
import html
from vroom.NER import (
    read_file,
    chunk_text_by_sentence,
    get_entities_from_file,
    get_positions_of_entities,
    tag_text_with_entities,
)
import json
import re


def submission(
    name_exp: str = "GPT-3_NER_chunks_determinant", baseline_fuzzy=False
):
    """ This function aims to generate the NER chunks with GPT.

     Basically, we use the GPT model to generate from a given text all
     the person entities. We then use a self_verification step in which
     a GPT model is asked to verify if a given person entity is indeed
     a person entity. If the model answers yes, we keep the entity, if
     it answers no, we discard it. Then, we use the remaining entities
     to generate the cooccurrences graph thanks to a fuzzy partial
     token matching.

     The files are saved along the way to avoid recomputing everything
     if the script is stopped.

     Make sure to modify the paths to the data and the experiment name
     to avoid overwriting the files.

     Args:
          None

     Returns:
          None
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

            experiment_name = os.path.join(
                "save", "kaggle", book_code, name_exp
            )
            save_path = os.path.join(
                experiment_name, "ner", f"chapter_{chapter}.json"
            )
            graph_manager = GraphManager()

            if os.path.exists(save_path):
                print("Already proceed NER GPT: ", path)
            else:
                print("Creating NER GPT...")
                logger = JSONLogger(save_path)
                generate_GPT_NER(path, logger)

            output_path = os.path.join(
                experiment_name,
                "verif_reviewed",
                f"chapter_{chapter}_verif.json",
            )
            if os.path.exists(output_path):
                print("Already proceed self_verification: ", output_path)
                entities = get_data_from_json(output_path)
            else:
                print("Creating self_verification...")
                logger = JSONLogger(output_path)
                entities = self_verification(path, save_path, logger)

            output_path = os.path.join(
                experiment_name,
                "cooocurrences",
                f"chapter_{chapter}_coocurrences.json",
            )

            print("Creating coocurrences...")
            logger = JSONLogger(output_path)

            if baseline_fuzzy:
                coocurrences = get_coocurrences_GPT_ner_fuzzy(
                    path, entities, logger
                )
            else:
                coocurrences = get_cooccurences_with_aliases_and_gpt(
                    path, entities, logger
                )

            graph_manager.add_cooccurrences(coocurrences)
            df_dict["ID"].append(f"{book_code}{chapter-1}")
            df_dict["graphml"].append(
                "".join(
                    html.unescape(s) if isinstance(s, str) else s
                    for s in graph_manager.generate_graph()
                )
            )

    print("Saving the submission file...")
    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("submission.csv")
    print("Done !")


def generate_GPT_NER(txt_path, logger=None):
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

     Input : C’est à mi-tirade que Seldon comprit qu’il s’adressait à l’empereur Cléon, premier du nom, ce qui lui coupa la respiration. Il y avait effectivement un vague faux air de ressemblance, maintenant qu’il y regardait de plus près, avec l’hologramme officiel que l’on voyait constamment aux informations, mais sur ces portraits, Cléon

     Output : C’est à mi-tirade que @@Seldon##  comprit qu’il s’adressait à @@l’empereur Cléon##, premier du nom, ce qui lui coupa la respiration. Il y avait effectivement un vague faux air de ressemblance, maintenant qu’il y regardait de plus près, avec l’hologramme officiel que l’on voyait constamment aux informations, mais sur ces portraits, @@Cléon##

     Je ne dévierais pas de cette tâche et ferais exactement comme dans les exemples.
     Si je ne trouve pas de Personnage, je ne mettrais pas de balise. Si je croise un personnage, avec de la ponctuation, je mettrais la balise
     avant la ponctuation sauf si c'est un mot composé avec des tirets. Garde les determinants avec le nom d'un personnage si tu peux.
     """
    content = read_file(txt_path)
    chunks = chunk_text_by_sentence(content, batch_size=4)
    experiment_details = """
     gpt-4-1106-preview + 4 sentences per batch
     """

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-3.5-turbo-1106",  # gpt-4-1106-preview
    }
    gpt_outputs = []

    for i, chunk in enumerate(chunks):
        print("*" * 50)
        print(chunk)
        user_prompt = f"""

          Input : {chunk}

          Output : """

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
        print("*" * 50)
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


def self_verification_no_json(txt_path, entities, logger=None):
    """ This function aims to verify if the NER chunks are indeed
     person entities.

     Basically, we use the GPT model to generate from a given text all
     the person entities. We then use a self_verification step in which
     a GPT model is asked to verify if a given person entity is indeed
     a person entity. If the model answers yes, we keep the entity, if
     it answers no, we discard it.

     Args:
          txt_path (str): path to the text file
          json_saved_ner_chunks_path (str): path to the json file
          logger (Logger): logger to save the data

     Returns:
          entities (list): list of the person entities
     """
    client = OpenAI()

    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-4-1106-preview",  # gpt-4-1106-preview , gpt-3.5-turbo-1106
    }

    system_prompt = r"""
     La tâche consiste à vérifier si le mot est une entité de personnage extraite de la phrase donnée.   Voici quelques exemples :

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "CLÉON Ier" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "Empereur" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "Empereur" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Par conséquent, peu importe que la prédiction de l’avenir soit ou non une réalité, n’est-ce pas ? Si un mathématicien devait me prédire un règne long et heureux, et pour l’Empire une ère de paix et de prospérité... eh bien, ne serait-ce pas une bonne chose ? — Ce serait assurément agréable à entendre, mais ça nous avancerait à quoi, Sire ? — Eh bien, si les gens croyaient ça, ils agiraient certainement selon cette croyance.

     Question : Le mot "Sire" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse :  Oui

     Phrase : J’ai appris qu’on vous avait vu en compagnie d’un garde impérial, vous dirigeant vers la porte du Palais. Vous n’auriez pas, par le plus grand des hasards, été reçu par l’Empereur, non ? » Le sourire déserta le visage de Seldon. C’est avec lenteur qu’il répondit : « Si tel avait été le cas, ce ne serait certes pas un sujet que je confierais pour publication.

     Question : Le mot "garde impérial" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse :  Non
     """

    content = read_file(txt_path)
    chunks = chunk_text_by_sentence(content, batch_size=5)

    labeled_gpt_entities = entities
    init_gpt_entities = labeled_gpt_entities.copy()

    # self verification
    checked_entities = []
    save_prompts_and_responses = []
    for chunk in chunks:
        for entity in labeled_gpt_entities:
            if entity in chunk and entity not in checked_entities:

                user_prompt = f"""

                    Phrase : {chunk}

                    Question : Le mot "{entity}" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

                    Réponse :
                    """

                response = client.chat.completions.create(
                    **params,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                generated_content = response.choices[0].message.content

                if generated_content.lower() == "oui":
                    checked_entities.append(entity)
                else:
                    labeled_gpt_entities.remove(entity)

                save_prompts_and_responses.append(
                    {"prompt": user_prompt, "response": generated_content}
                )

    saves = {}
    saves["final_gpt_entities"] = checked_entities
    saves["init_gpt_entities"] = init_gpt_entities
    saves["prompts_and_responses"] = save_prompts_and_responses

    if logger is not None:
        logger(saves)

    return saves


def find_word_positions(word_list, input_text):
    positions = []

    for word in word_list:
        start = input_text.find(word)
        while start != -1:
            end = start + len(word)
            positions.append({"word": word, "start": start, "end": end})
            start = input_text.find(word, start + 1)

    # Sort the list of dictionaries based on the "start" key
    sorted_positions = sorted(positions, key=lambda x: x["start"])

    return sorted_positions


def get_cooccurences_with_aliases_and_gpt(
    path: str, entities, logger: JSONLogger = None
):
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
     '1' : ['Hari Seldon', 'Seldon'],
     '2' : ['Lieutenant Alban Wellis', 'Wellis'],
     '3' : ['Hummin'],
     '4' : ['Eto Demerzel', 'Demerzel']
     }

     Tu vas desormais recevoir une liste de personnes en input, et tu dois les regrouper s'ils se réfèrent à la même personne. Tu n'as pas le droit d'inventer de nouveaux personnages et tu dois uniquement utiliser cette liste. Les noms ne doivent pas être modifiés, y compris les espaces, apostrophes, etc. Si tu ne respectes pas ces règles, tu seras fortement pénalisé.
     Tu as le droit de supprimer des personnes si tu penses qu'ils ne sont pas des personnages du livre. En effet, le système qui prédit un personnage peut se tromper, et tu dois le corriger.

     Fais-le pour les personnes suivantes et essaie de trouver les meilleurs regroupements possibles. Je compte sur toi, merci !
     """

    gpt_entities = entities["final_gpt_entities"]
    print("entities = ", gpt_entities)
    input_text = read_file(path)
    positions = find_word_positions(gpt_entities, input_text)
    cooccurences = get_cooccurences([input_text], [positions])
    word_entities = set(gpt_entities)
    user_prompt = """

     Input :
     """
    user_prompt += "\n".join(set(word_entities)) + "\nOutput : "
    experiment_details = """
     gpt-4-1106-preview avec 1 exemples de donné en prompt.
     """  # GPT-3.5-turbo-1106

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-4-1106-preview",  # gpt-4-1106-preview, gpt-3.5-turbo-1106
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

    print("aliases = ", aliases)
    cooc = find_cooccurences_aliases(cooccurences, aliases)
    return cooc


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

    entities_unfold = [entity for sublist in entities for entity in sublist]

    cooccurences = get_cooccurences(chunks, entities)

    # remove entities in entities_unfold that are not in entities
    entities_unfold = [
        entity
        for entity in entities_unfold
        if entity["word"] in entities["final_gpt_entities"]
    ]
    aliases = get_aliases_fuzzy_partial_token(entities_unfold, 99)

    if logger is not None:
        saves = {}
        for i, (chunk, entity) in enumerate(zip(chunks, entities)):
            saves[f"chunk_{i}"] = {
                "chunk": chunk,
                "entities": entity,
            }
        saves["entities"] = list(
            set([entity["word"] for entity in entities_unfold])
        )
        saves["aliases"] = aliases
        saves["cooccurences"] = cooccurences
        logger(saves)

    cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
    return cooccurences_aliases


def self_verification(txt_path, json_saved_ner_chunks_path, logger=None):
    """ This function aims to verify if the NER chunks are indeed
     person entities.

     Basically, we use the GPT model to generate from a given text all
     the person entities. We then use a self_verification step in which
     a GPT model is asked to verify if a given person entity is indeed
     a person entity. If the model answers yes, we keep the entity, if
     it answers no, we discard it.

     Args:
          txt_path (str): path to the text file
          json_saved_ner_chunks_path (str): path to the json file
          logger (Logger): logger to save the data

     Returns:
          entities (list): list of the person entities
     """
    client = OpenAI()

    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-3.5-turbo-1106",  # gpt-4-1106-preview , gpt-3.5-turbo-1106
    }

    system_prompt = r"""
     La tâche consiste à vérifier si le mot est une entité de personnage extraite de la phrase donnée.   Voici quelques exemples :

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "CLÉON Ier" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "Empereur" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Mathématicien CLÉON Ier— ... dernier Empereur galactique de la
          dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
          année que Hari Seldon.

     Question : Le mot "Empereur" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse : Oui

     Phrase : Par conséquent, peu importe que la prédiction de l’avenir soit ou non une réalité, n’est-ce pas ? Si un mathématicien devait me prédire un règne long et heureux, et pour l’Empire une ère de paix et de prospérité... eh bien, ne serait-ce pas une bonne chose ? — Ce serait assurément agréable à entendre, mais ça nous avancerait à quoi, Sire ? — Eh bien, si les gens croyaient ça, ils agiraient certainement selon cette croyance.

     Question : Le mot "Sire" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse :  Oui

     Phrase : J’ai appris qu’on vous avait vu en compagnie d’un garde impérial, vous dirigeant vers la porte du Palais. Vous n’auriez pas, par le plus grand des hasards, été reçu par l’Empereur, non ? » Le sourire déserta le visage de Seldon. C’est avec lenteur qu’il répondit : « Si tel avait été le cas, ce ne serait certes pas un sujet que je confierais pour publication.

     Question : Le mot "garde impérial" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

     Réponse :  Non
     """

    content = read_file(txt_path)
    chunks = chunk_text_by_sentence(content, batch_size=5)

    with open(json_saved_ner_chunks_path, "r") as f:
        gpt_output = json.load(f)

    tagged_text_gpt = []

    for key in gpt_output:
        for values in gpt_output[key]:
            if values == "ner_chunk":
                tagged_text_gpt.append(gpt_output[key][values])

    labeled_gpt_entities = []
    for chunk in tagged_text_gpt:
        # extract all entities between @@ and ##
        entities = re.findall(r"@@(.*?)##", chunk)
        labeled_gpt_entities.append(entities)
    # flatten list
    labeled_gpt_entities = [
        item for sublist in labeled_gpt_entities for item in sublist
    ]
    labeled_gpt_entities = list(set(labeled_gpt_entities))
    init_gpt_entities = labeled_gpt_entities.copy()

    # self verification
    checked_entities = []
    save_prompts_and_responses = []
    for chunk in chunks:
        for entity in labeled_gpt_entities:
            if entity in chunk and entity not in checked_entities:

                user_prompt = f"""

                    Phrase : {chunk}

                    Question : Le mot "{entity}" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

                    Réponse :
                    """

                response = client.chat.completions.create(
                    **params,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                generated_content = response.choices[0].message.content

                if generated_content.lower() == "oui":
                    checked_entities.append(entity)
                else:
                    labeled_gpt_entities.remove(entity)

                save_prompts_and_responses.append(
                    {"prompt": user_prompt, "response": generated_content}
                )

    saves = {}
    saves["final_gpt_entities"] = checked_entities
    saves["init_gpt_entities"] = init_gpt_entities
    saves["prompts_and_responses"] = save_prompts_and_responses

    if logger is not None:
        logger(saves)

    return saves


def get_data_from_json(json_path):
    """ Get data from json file

     Args:
          json_path (str): path to json file

     Returns:
          dict: data from json file
     """
    with open(json_path, "r") as f:
        output = json.load(f)
    return output


def get_coocurrences_GPT_ner_fuzzy(input_file, entities, logger=None):
    """ Get cooccurences of entities in a text

     Args:
          input_file (str): path to text file
          entities (list): list of entities to find cooccurences
          logger (function, optional): function to save data. Defaults to None.

     Returns:
          dict: cooccurences of entities
     """
    gpt_entities = entities["final_gpt_entities"]
    print("entities = ", gpt_entities)
    input_text = read_file(input_file)
    positions = find_word_positions(gpt_entities, input_text)
    cooccurences = get_cooccurences([input_text], [positions])
    gpt_entities = [{"word": entity} for entity in gpt_entities]
    aliases = get_aliases_fuzzy_partial_token(gpt_entities, 99)
    cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
    print("cooccurences_aliases = ", cooccurences_aliases)

    save = {
        "cooccurences": cooccurences,
        "entities": gpt_entities,
        "aliases": aliases,
        "cooccurences_aliases": cooccurences_aliases,
    }
    if logger is not None:
        logger(save)
    return cooccurences_aliases


def get_coocurrences_GPT_ner_GPT(input_file, entities, logger=None):
    """ Get cooccurences of entities in a text

     Args:
          input_file (str): path to text file
          entities (list): list of entities to find cooccurences
          logger (function, optional): function to save data. Defaults to None.

     Returns:
          dict: cooccurences of entities
     """
    system_prompt = r"""
     Je suis un excellent linguiste.
     La tâche consiste à regrouper les entités de type "Personnes" dans la liste de personnes données. Certaines personnes peuvent-être des références à une entité commune.
     À partir de tes connaissances linguistiques, tu dois déterminer qui est qui.

     Tu devras donner ta réponse sous le format JSON suivant :

     {
     '0' : [reference_1, …],
     '1' : [reference_1, …],
     }

     Chaque entrée du JSON correspond à un personnage et à l'ensemble de ses références. La clef est un chiffre qui donne la position dans le JSON. La position n’a pas d’importance.

     Voici des exemples pour t'aider :

     Liste de personnes :

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

     Réponse :

     {
     '0' : ['CLÉON Ier', 'Empereur', 'Cléon', 'Sire', 'l'Empereur', 'l'empereur Cléon'],
     '1' : ['Hari Seldon', 'Seldon', 'Eto Demerzel', 'Demerzel'],
     '2' : ['Lieutenant Alban Wellis', 'Wellis'],
     '3' : ['Hummin'],
     }

     Liste de personnes :

     CLÉON Ier
     Cléon
     Seldon
     Hari Seldon
     Hari
     Eto Demerzel
     Demerzel
     Goutte-de-Pluie Quarante-trois
     Goutte-de-Pluie Quarante-cinq
     Dors
     Dors Seldon

     Réponse :

     {
     '0' : ['CLÉON Ier', 'Cléon'],
     '1' : ['Seldon', 'Hari Seldon', 'Hari'],
     '2' : ['Eto Demerzel', 'Demerzel'],
     '3' : ['Goutte-de-Pluie Quarante-trois'],
     '4' : ['Goutte-de-Pluie Quarante-cinq'],
     '5' : ['Dors', 'Dors Seldon'],
     }
     """
    gpt_entities = entities["final_gpt_entities"]

    out = "\n".join(gpt_entities)
    user_prompt = f"""
     print("Liste de personnes : {out}")
     Liste de personnes :

     {out}

     Réponse :
     """

    client = OpenAI()
    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-4-1106-preview",  # gpt-4-1106-preview
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

    tagged_file = tag_text_with_entities(input_file, gpt_entities)
    positions = [get_positions_of_entities(tagged_file)]
    cooccurences = get_cooccurences([tagged_file], positions)
    gpt_entities = [{"word": entity} for entity in gpt_entities]
    cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
    save = {
        "cooccurences": cooccurences,
        "entities": gpt_entities,
        "aliases": aliases,
        "cooccurences_aliases": cooccurences_aliases,
    }
    save["gpt"] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "generated_content": generated_content,
        "params": params,
    }
    if logger is not None:
        logger(save)
    return cooccurences_aliases


if __name__ == "__main__":
    name_exp = "GPT_4_NER"
    submission(name_exp, baseline_fuzzy=False)
