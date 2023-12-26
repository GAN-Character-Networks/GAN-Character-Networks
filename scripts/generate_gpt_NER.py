r""" A class for loading and infering to generative large language models.

Authors
--------
 * Adel Moumen 2023
"""

from openai import OpenAI
import os
import glob
from vroom.NER import chunk_text, read_file
import json 

client = OpenAI()


# output_file_path = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.gpt.v1.labeled"
input_file = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled")
output_file = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.gpt.ner.v4.labeled.json")
content = read_file(input_file)
chunks = chunk_text(content, 500)

output = ""

system_prompt = """
Tu es un extracteur d'entités. 
Ton but est d'extraire tous les personnages du livre de science-fiction 'Le Cycle des Fondations' d'Isaac Asimov. 
Tu verras des passages du livre que tu devras utiliser. Dans ta définition, un personnage est un individu qui apparaît dans un passage du livre. 
Ce personnage peut être uniquement cité par son nom ou être très actif dans la discussion. 
Pour réaliser cette tâche, je souhaite que tu me retournes la liste des personnages que tu rencontres. 
Tu dois me donner l'ensemble des entités. Tu n'as pas le droit de modifier le nom des personnages ou d'en inventer de nouveaux, utilise seulement le texte. 
Tu peux avoir plusieurs références d'un même personnage, renvoie l'ensemble des références. 
Par exemple, Hari Seldon est souvent appelé Hari et/ou Seldon. Je veux que tu listes aussi cela. 
Le retour doit être fait dans un JSON. Fait attention à ne pas capturer des noms de personnages qui ne sont pas des personnages comme par exemple
des noms de planètes, des fonctions (ministre), etc.

Pour t'aider, voici la liste officielle des personnages du livre :
<start>
Arcadia Darell
Agis XIV
Ammel Brodrig
Bail Channis
Bayta Darell
Bel Arvardan
Bel Riose
Chetter Hummin
Cléon Ier
Cléon II
Dagobert IX
Dame Callia
Dors Venabili
Ducem Barr
Ebling Mis
Elijah Baley
Eskel Gorov
Eto Demerzel
Fallom
Gaal Dornick
Golan Trevize
Han Pritcher
Hari Seldon
Harlan Branno
Hober Mallow
Homir Munn
Indbur III
Janov Pelorat
Joie
Jole Turbor
Jord Commasson
Lathan Devers
Lepold Ier
Lev Meirus
Lewis Pirenne
Le Mulet
Limmar Ponyets
Munn Li Compor
Novi Sura
Pelleas Anthor
Preem Palver
Quindor Shandess
R. Daneel Olivaw
Raych Seldon
Salvor Hardin
StettinStor Gendibal
Toran Darell
Toran Darell II
Wanda Seldon
Wienis
Yugo Amaryl
<end>

Voici des exemples : 

Exemple 1 : 

Texte : 
<start>
                           Mathématicien


     CLÉON Ier— ... dernier Empereur galactique de la
dynastie Entun. Né en l’an 11988 de l’Ère Galactique, la même
année que Hari Seldon. (On pense que la date de naissance de
Seldon, que certains estiment douteuse, aurait pu être
« ajustée » pour coïncider avec celle de Cléon que Seldon, peu
après son arrivée sur Trantor, est censé avoir rencontré.)
     Cléon est monté sur le trône impérial en 12010, à l’âge de
vingt-deux ans, et son règne représente un étrange intervalle
de calme dans ces temps troublés. Cela est dû sans aucun doute
aux talents de son chef d’état-major, Eto Demerzel, qui sut si
bien se dissimuler à la curiosité médiatique que l’on a fort peu
de renseignements à son sujet. La psychohistoire nous apprend
bien des choses.
     Cléon, quant à lui...
                                       ENCYCLOPAEDIA GALACTICA2
<end>

Output : 

{
  'personnages': [ 'CLÉON Ier', 'Empereur', 'Hari Seldon', 'Seldon' , 'Cléon', 'Eto Demerzel']
}

Exemple 2 : 

Texte : 
<start>
     Étouffant un léger bâillement, Cléon demanda :
« Demerzel, auriez-vous, par hasard, entendu parler d’un
certain Hari Seldon ? »
     Cléon était empereur depuis dix ans à peine et, quand le
protocole l’exigeait, il y avait des moments où, pourvu qu’il fût
revêtu des atours et ornements idoines, il réussissait à paraître
majestueux. Il y était arrivé, par exemple, pour son portrait
<end>

Output : 

{
  'personnages': ['Cléon', 'Demerzel', 'Hari Seldon', 'Cléon']
}

Exemple 3 : 

Texte : 
<start>
     2 Toutes les citations de l'Encyclopaedia Galactica reproduites ici
proviennent de la 116e édition, publiée en 1020 E.F. par la Société
d’édition de l'Encyclopaedia Galactica, Terminus, avec l'aimable
autorisation des éditeurs.
semblerait malgré tout qu’il puisse encore arriver des choses
intéressantes. Du moins, à ce que j’ai entendu dire.
     — Par le ministre des Sciences ?
     — Effectivement. Il m’a appris que ce Hari Seldon  a assisté
à un congrès de mathématiciens ici même, à Trantor – ils
l’organisent tous les dix ans, pour je ne sais quelle raison ; il
aurait démontré qu’on peut prévoir mathématiquement
l’avenir. »
<end>

Output : 

{
  'personnages': ['Hari Seldon']
}

Exemple 4 : 

Texte : 
<start>
     — Je le crois bien, Sire  », répondit Demerzel . Ses yeux
scrutaient attentivement l’Empereur , comme pour voir jusqu’où
il pouvait se permettre d’aller. « Pourtant, s’il devait en être
ainsi, n’importe qui pourrait prophétiser. Le ministre n'a-t-il
pas pensé à cela ?
<end>

Output : 

{
  'personnages': ['Sire', 'Demerzel', 'Empereur']
}


Fait le pour l'exemple suivant.

Texte :

     Demerzel se permit un petit sourire. « Ou le ministre des
Sciences, homme sans grande jugeote, a été induit en erreur, ou
ce mathématicien s’est trompé. Il ne fait aucun doute que cette
histoire de prédiction de l’avenir relève d’un puéril rêve de
magie.

Output : 
<start>
"""
entities = []
for i, chunk in enumerate(chunks):
    print("chunk = ", chunk)
    
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-1106", # gpt-4-1106-preview / gpt-3.5-turbo-1106
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk + "\n<end>"},
      ],
      seed=42,
      temperature=0,
#       presence_penalty=-2,
      response_format={ "type": "json_object" },
    )

    generated_content = response.choices[0].message.content
    generated_content = json.loads(generated_content)
    print()
    print("content = ", generated_content)

    entities += generated_content["personnages"]
    print("total entities = ", set(entities))
    print('*' * 100)



print()
print('*' * 100)
print("FINAL")
print("total entities = ", set(entities))

with open(output_file, "w") as f:
    # write entities as json
     json.dump({"personnages": list(set(entities))}, f, indent=4)

