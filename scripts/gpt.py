r""" A class for loading and infering to generative large language models.

Authors
--------
 * Adel Moumen 2023
"""

from openai import OpenAI
import os
import glob
import json
from vroom.NER import write_bio_tag_file


system_prompt = r"""
Vous êtes un expert capable d'extraire avec précision les interactions entre les personnages d'un texte donné. Le texte comprend des balises de personnage marquées avec <B-PER> pour le début et <I-PER> pour les parties intermédiaires du nom du personnage. Vous devez identifier et fournir une représentation des interactions sous le format : <personnage 1 ; personnage 2>. Par exemple, "Romain <B-PER> est le frère de Paul <B-PER>", la réponse attendue est <Romain;Paul>.  Je ne veux qu'uniquement la liste des interactions. Fait le pour ce texte :
"""

client = OpenAI()

directory_path = 'data/kaggle/les_cavernes_d_acier'
output_folder = 'data/output/NER+GPT3/les_cavernes_d_acier/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

txt_files = glob.glob(os.path.join(directory_path, '*'))

for txt_file in txt_files:
    output_file_path = os.path.join(output_folder, "text_augmented_with_ner", os.path.basename(txt_file))
    
    write_bio_tag_file(
      txt_file, 
      output_file_path
    )

    with open(output_file_path, 'r') as file:
        content = file.read()
        # Process the content of the file as needed
        print(f"Processing {output_file_path}")

        response = client.chat.completions.create(
          model="gpt-3.5-turbo-16k",
          messages=[
            {"role": "system", "content": system_prompt},
            #{"role": "user", "content": system_prompt + content},
            {"role": "user", "content": content},
          ]
        )
        generated_content = response.choices[0].message.content

        output_file_path = os.path.join(output_folder, "final_text", os.path.basename(txt_file))
        
        if not os.path.exists(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))

        if not output_file_path.lower().endswith('.txt'):
            output_file_path += '.txt'

        with open(output_file_path, 'w') as output_file:
            output_file.write(generated_content)
