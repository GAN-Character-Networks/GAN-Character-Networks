r""" Submission file for using GPT to generate the NER chunks + GPT self_verification + fuzzy partial token matching.

Authors
--------
 * Adel Moumen 2023
"""

from vroom.baseline import *
from vroom.loggers import JSONLogger
from vroom.GraphManager import GraphManager
from tqdm import tqdm
import os
import pandas as pd
import html

def submission(name_exp: str = "GPT-3_NER_chunks_determinant"):
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
               
               experiment_name = os.path.join("save", "kaggle", book_code, name_exp)
               save_path = os.path.join(experiment_name, "ner", f"chapter_{chapter}.json")
               graph_manager = GraphManager()

               if os.path.exists(save_path):
                    print("Already proceed NER GPT: ", path)
               else:
                    logger = JSONLogger(save_path)
                    generate_GP_NER(path, logger)

               output_path = os.path.join(experiment_name, "verif", f"chapter_{chapter}_verif.json")
               if os.path.exists(output_path):
                    print("Already proceed self_verification: ", output_path)
                    entities = get_data_from_json(output_path)
               else:
                    logger = JSONLogger(output_path)
                    entities = self_verification(path, save_path, logger)

               output_path = os.path.join(experiment_name, "cooocurrences", f"chapter_{chapter}_coocurrences.json")
               
               if os.path.exists(output_path):
                    print("Already proceed coocurrences: ", output_path)
                    coocurrences = get_data_from_json(output_path)
                    coocurrences = coocurrences["cooccurences"]
               else:
                    print("Creating coocurrences...")
                    logger = JSONLogger(output_path)
                    coocurrences = get_coocurrences_GPT_ner(path, entities, logger)
               graph_manager.add_cooccurrences(coocurrences)
               df_dict["ID"].append(f"{book_code}{chapter-1}")
               df_dict["graphml"].append("".join(html.unescape(s) if isinstance(s, str) else s for s in graph_manager.generate_graph()))
               
     df = pd.DataFrame(df_dict)
     df.set_index("ID", inplace=True)
     df.to_csv("submission.csv")

def generate_GP_NER(txt_path, logger = None):
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
     content = read_file(txt_path)
     chunks = chunk_text_by_sentence(content, batch_size = 4) 
     experiment_details = """
     gpt-4-1106-preview + 4 sentences per batch
     """

     client = OpenAI()
     params = {
          "temperature": 0,
          "seed": 42,
          "model": "gpt-4-1106-preview", # gpt-4-1106-preview
     }
     gpt_outputs = []
     
     for i, chunk in enumerate(chunks):
          print('*' * 50)
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

def self_verification(txt_path, json_saved_ner_chunks_path, logger = None):
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
          "model": "gpt-4-1106-preview", # gpt-4-1106-preview
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

     content = read_file(txt_path)
     chunks = chunk_text_by_sentence(content, batch_size = 5) 

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
          entities = re.findall(r'@@(.*?)##', chunk)
          labeled_gpt_entities.append(entities)
     
     # flatten list
     labeled_gpt_entities = [item for sublist in labeled_gpt_entities for item in sublist]
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

                    save_prompts_and_responses.append({
                         "prompt": user_prompt,
                         "response": generated_content
                    })

     save_prompts_and_responses["final_gpt_entities"] = checked_entities
     save_prompts_and_responses["init_gpt_entities"] = init_gpt_entities

     if logger is not None:
          logger(save_prompts_and_responses)

     return save_prompts_and_responses

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

def get_coocurrences_GPT_ner(input_file, entities, logger = None):
     """ Get cooccurences of entities in a text

     Args:
          input_file (str): path to text file
          entities (list): list of entities to find cooccurences
          logger (function, optional): function to save data. Defaults to None.

     Returns:
          dict: cooccurences of entities
     """
     entities = entities["final_gpt_entities"]
     tagged_file = tag_text_with_entities(input_file, entities)
     positions = [get_positions_of_entities(tagged_file)]
     cooccurences = get_cooccurences([tagged_file], positions)
     entities = [{"word": entity} for entity in entities]
     aliases = get_aliases_fuzzy_partial_token(entities, 99)
     
     cooccurences_aliases = find_cooccurences_aliases(cooccurences, aliases)
     
     save = {
          "cooccurences": cooccurences,
          "entities": entities,
          "aliases": aliases,
          "cooccurences_aliases": cooccurences_aliases
     }
     if logger is not None:
          logger(save)
     return cooccurences_aliases
     

if __name__ == "__main__":
    name_exp = "GPT_4_NER_self_verification"
    submission(name_exp)