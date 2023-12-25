from openai import OpenAI
from vroom.NER import *
import os
import tiktoken
from vroom.GPTTokenizer import GPTTokenizer
import json

unlabeled_chapter = os.path.join("../", "data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled")

entities, chunks = get_entities_from_file(unlabeled_chapter)

all_entities_names = []
for chunk in entities: 
    all_entities_names += [
        entity["word"] for entity in chunk
    ]
entities = set(all_entities_names)

text_tagged = tag_text_with_entities(unlabeled_chapter, entities)

# Appel à l'API de GPT-3 pour générer les alias
OPENAI_API_KEY="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"

system_prompt = """
Tu es un expert dans l'étude des personnages dans les romans. J'ai un texte annoté par un modèle d'NER, et je n'ai gardé que les entités qu'il a estimé être des personnages. Elles sont situées entre les balises <PER></PER>. J'ai besoin qu'à partir de ce texte, tu me construises un JSON qui compile tous les personnages et leurs alias associé. Tu dois le faire comme dans cet exemple : 

Input : 
<start>
Mathématicien  <PER> CLÉON Ier </PER>  — ... dernier  <PER> Empereur </PER>  galactique de la dynastie Entun. Né en l'an 11988 de l'Ère Galactique, la même année que  <PER> Hari  Seldon  </PER> . (On pense que la date de naissance de  <PER> Seldon </PER> , que certains estiment douteuse, aurait pu être « ajustée » pour coïncider avec celle de  <PER> Cléon </PER>  que  <PER> Seldon </PER>
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

client = OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-3.5-turbo-1106" # gpt-4-1106-preview

len_chunk = 5000
tokenizer = GPTTokenizer(model)
len_prompt = len(tokenizer.tokenize(system_prompt))

chunks = chunk_text(text_tagged, len_chunk)

json_entities = []

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
    print("Tokens sent : ", len(tokenizer.tokenize(chunk+system_prompt)))
    print("Tokens received : ", len(tokenizer.tokenize(response.choices[0].message.content)))
    print("Text computed : ", chunk)
    print("Text received : ", response.choices[0].message.content)


json_entities = [json.loads(json_entity) for json_entity in json_entities]

all_entities = {}
for json_entity in json_entities:
    for entity in json_entity:
        if entity in all_entities:
            all_entities[entity]["aliases"] += json_entity[entity]["aliases"]
        else:
            all_entities[entity] = json_entity[entity]

for entity in all_entities:
    all_entities[entity]["aliases"] = list(set(all_entities[entity]["aliases"]))
    all_entities[entity]["aliases"].append(entity)


filtering_prompt = """
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

print("Tokens sent : ", len(tokenizer.tokenize(system_prompt)))
print("Tokens received : ", len(tokenizer.tokenize(response.choices[0].message.content)))
print("Text computed : ", chunk)
print("Text received : ", response.choices[0].message.content)

# Ecrire le json dans un fichier
with open("../data/test_set/prelude_a_fondation/chapter_1.gpt.json", "w") as f:
    json_entities = json.loads(response.choices[0].message.content)
    json.dump(json_entities, f, indent=4, ensure_ascii=False)