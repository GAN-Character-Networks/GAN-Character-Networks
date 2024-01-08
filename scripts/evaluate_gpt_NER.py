import json
import os
import re

from openai import OpenAI

from vroom.NER import (
    chunk_text_by_sentence,
    get_positions_of_entities,
    read_file,
    tag_text_with_entities,
)

unlabeled_chapter = os.path.join(
    "data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled"
)
labeled_chapter = os.path.join(
    "data", "test_set", "prelude_a_fondation", "chapter_1.labeled"
)
gpt_output_json = os.path.join(
    "save", "kaggle", "paf", "GPT-3_NER_chunks_determinant", "chapter_1.json"
)
content = read_file(unlabeled_chapter)
chunks = chunk_text_by_sentence(content, batch_size=5)
do_self_verification = True

with open(gpt_output_json, "r") as f:
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

labeled_gpt_entities = [
    item for sublist in labeled_gpt_entities for item in sublist
]
labeled_gpt_entities = list(set(labeled_gpt_entities))
print(labeled_gpt_entities)

if do_self_verification:
    client = OpenAI()

    params = {
        "temperature": 0,
        "seed": 42,
        "model": "gpt-3.5-turbo-1106",  # gpt-4-1106-preview
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

    Question : Le mot "Hari Seldon" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

    Réponse : Oui

    Phrase : Par conséquent, peu importe que la prédiction de l’avenir soit ou non une réalité, n’est-ce pas ? Si un mathématicien devait me prédire un règne long et heureux, et pour l’Empire une ère de paix et de prospérité... eh bien, ne serait-ce pas une bonne chose ? — Ce serait assurément agréable à entendre, mais ça nous avancerait à quoi, Sire ? — Eh bien, si les gens croyaient ça, ils agiraient certainement selon cette croyance.

    Question : Le mot "Sire" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

    Réponse :  Oui

    Phrase : J’ai appris qu’on vous avait vu en compagnie d’un garde impérial, vous dirigeant vers la porte du Palais. Vous n’auriez pas, par le plus grand des hasards, été reçu par l’Empereur, non ? » Le sourire déserta le visage de Seldon. C’est avec lenteur qu’il répondit : « Si tel avait été le cas, ce ne serait certes pas un sujet que je confierais pour publication.

    Question : Le mot "garde impérial" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

    Réponse :  Non
    """

    # self verification
    checked_entities = []
    for chunk in chunks:
        for entity in labeled_gpt_entities:
            if entity in chunk and entity not in checked_entities:

                user_prompt = f"""

                Phrase : {chunk}

                Question : Le mot "{entity}" dans la phrase d'entrée est-il une entité de personnage ? Veuillez répondre par Oui ou par Non.

                Réponse :
                """

                print(user_prompt)

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
                print("--------------------------------------------------")

                if generated_content == "Oui":
                    checked_entities.append(entity)
                else:
                    labeled_gpt_entities.remove(entity)
# else:
#    labeled_gpt_entities = ['Eto Demerzel', 'Demerzel', 'CLÉON Ier', 'Empereurs', 'Lieutenant Alban Wellis', 'empereur Cléon', 'Seldon', 'Empereur', 'Hummin', 'Sire', 'Hari Seldon', 'Wellis', 'Cléon']

print(labeled_gpt_entities)

labeled_gpt_entities += ["l’Empereur"]

unlabeled_text_tagged = tag_text_with_entities(
    unlabeled_chapter, labeled_gpt_entities
)
labeled_text_tagged = read_file(labeled_chapter)

positions_ner = get_positions_of_entities(unlabeled_text_tagged)
true_position = get_positions_of_entities(labeled_text_tagged)


def evaluate_ner(positions_ner, true_position):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # check if the position of the entity is the same
    for entity in positions_ner:
        if entity in true_position:
            true_positives += 1
        else:
            false_positives += 1

    # check if the entity is missing
    for entity in true_position:
        if entity not in positions_ner:
            false_negatives += 1

    print("true_positive", true_positives)
    print("false_positive", false_positives)
    print("false_negative", false_negatives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = true_positives / (
        true_positives + false_positives + false_negatives
    )

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1_score)
    print("accuracy", accuracy)


evaluate_ner(positions_ner, true_position)
