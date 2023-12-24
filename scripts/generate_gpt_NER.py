r""" A class for loading and infering to generative large language models.

Authors
--------
 * Adel Moumen 2023
"""

from openai import OpenAI
import os
import glob
from vroom.NER import chunk_text, read_file

client = OpenAI()

input_file = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.unlabeled"
output_file_path = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.gpt.v1.labeled"

content = read_file(input_file)
chunks = chunk_text(content, 500)

output = ""

historic = "{}"
example = """
Le format JSON est le suivant, il peut y avoir plusieurs alias pour un personnage.
Voici des examples, contenant une entrée, le texte, et une sortie, le JSON attendu : 

Texte entrée : 

TRANTOR. — ... Elle n’est presque jamais décrite comme
un monde vu de l’espace. Depuis longtemps, l’inconscient
collectif la voit comme un monde de l’intérieur dont l’image est
celle de la ruche humaine vivant sous dôme. Pourtant, il
existait également un extérieur, et il nous reste encore des
hologrammes pris de l’espace qui le montrent plus ou moins en
détail (cf. figures 14 et 15). On remarquera que la surface des
dômes, l’interface de la vaste cité et de l’atmosphère qui la
surmonte, surface appelée à l’époque la « Couverture », est...

Sortie : 

{

}

Texte entrée :

ROBOT. — ... Terme employé dans les légendes antiques
de plusieurs mondes pour qualifier ce qu’on appelle plus
communément des « automates ». Les robots sont
généralement décrits comme faits de métal et d’apparence
humaine, mais l’on suppose que certains auraient été de nature
pseudo-organique. La croyance populaire veut qu’au cours de
la Fuite, Hari Seldon ait aperçu un véritable robot, mais la
véracité de cette anecdote reste douteuse. Nulle part, dans les
volumineux écrits laissés par Seldon, il n’est fait mention du
moindre robot, quoique...

                                  ENCYCLOPAEDIA GALACTICA

Sortie : 
{
    "Hari Seldon": [
        "Seldon"
    ]
}

Texte entrée :

Seldon réitéra son coup, une deuxième, puis une troisième
fois, mais cette fois le sergent Thalus, anticipant l’attaque,
abaissa l’épaule pour l’encaisser dans le gras du muscle.
      Dors avait sorti ses couteaux.
      « Sergent, lança-t-elle d’une voix forte. Tournez-vous dans
cette direction. Comprenez bien que je serai peut-être obligée de
vous blesser sérieusement si vous persistez à emmener le
docteur Seldon contre son gré. »

Sortie :
{
    "Seldon": [],
    "Thalus": []
}

"""
for i, chunk in enumerate(chunks):
    print("chunk = ", chunk)

    system_prompt_1 = r"""
    Tu es un expert en extration d'entités personnages des livres de science fiction 'Fondations' de Isaac Asimov. 
    
    Trouve toutes les entités personnages et toutes les façons dont ils sont mentionnés dans le texte et 
    donne ta réponse sous le format JSON.

    Il peut exister plusieurs personnages, et tu dois ajouter autant de lignes que de personnages dans le JSON. 
    """

    system_prompt_1 = system_prompt_1 + example

    system_prompt_2 = """
    Voici l'historique des précédents personnages et leur alias trouvés :

    Historique : 

    """ + historic + """
    
    Tu dois absolument réutiliser l'historique, et ajouter dedans les nouveaux personnages et alias trouvés. 
    Tu ne dois pas retirer des personnages de l'historique. Utilise uniquement ce qui est écrit litérallement dans le texte.

    Il se peut qu'il n'existe pas de personnages dans le texte donné. Si tel est le cas, tu dois renvoyer l'historique.

    Fait attention à ne pas supplémenter les entrées de l'historique. Si un personnage est dans l'historique, 
    mais n'apparait pas dans le texte, tu dois le laisser dans le json de sortie.

    Texte entrée : 
    """

    system_prompt = system_prompt_1 + system_prompt_2

    print("system_prompt = ", system_prompt)
    
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-1106", # gpt-4-1106-preview
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk},
      ],
      seed=42,
      temperature=0,
#       presence_penalty=-2,
      response_format={ "type": "json_object" },
    )

    generated_content = response.choices[0].message.content
    print()
    print("content = ", generated_content)
    historic = generated_content
    
    print('*' * 100)

    if i > 5:
        exit()
    
print("*" * 200)
print("generated_content = ", generated_content)
print()
print("historic = ", historic)


exit()
with open(output_file_path, "w") as f:
    f.write(output)
