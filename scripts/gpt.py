r""" A class for loading and infering to generative large language models.

Authors
--------
 * Adel Moumen 2023
"""

from openai import OpenAI
import os
import glob
import json

system_prompt = r"""
Tu es un as de l'extraction de personnages, chargé de repérer les protagonistes d'un texte et de préciser toute relation entre eux. Ton objectif : générer une liste de personnages suivant le format <personnage_1;personnage_2>. Par exemple, pour le texte "Romain est le frère de Paul", la réponse attendue est <Romain;Paul>.
Exécute cette mission avec le texte ci-dessous et assure-toi de formater ta réponse correctement. Je veux uniquement la liste des personnages, rien d'autre, exclusivement <personnage1; personnage2>, etc :
"""

client = OpenAI()

directory_path = 'data/kaggle/les_cavernes_d_acier'
folder_path = 'data/output/gpt3'

txt_files = glob.glob(os.path.join(directory_path, '*'))

for txt_file in txt_files:
    with open(txt_file, 'r') as file:
        content = file.read()
        # Process the content of the file as needed
        print(f"Content of {txt_file}:\n")

        response = client.chat.completions.create(
          model="gpt-3.5-turbo-16k",
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
          ]
        )
        generated_content = response.choices[0].message.content

        output_file_path = os.path.join(folder_path, os.path.basename(txt_file))
        
        if not output_file_path.lower().endswith('.txt'):
            output_file_path += '.txt'

        with open(output_file_path, 'w') as output_file:
            output_file.write(generated_content)