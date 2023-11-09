r""" A class for loading and infering to generative large language models.

Authors
--------
 * Adel Moumen 2023
"""

from openai import OpenAI

client = OpenAI()
system_prompt = r"""
Tu es un as de l'extraction de personnages, chargé de repérer les protagonistes d'un texte et de préciser toute relation entre eux. Ton objectif : générer une liste de personnages suivant le format <personnage_1;personnage_2>. Par exemple, pour le texte "Romain est le frère de Paul", la réponse attendue est <Romain;Paul>.
Exécute cette mission avec le texte ci-dessous et assure-toi de formater ta réponse correctement. Je veux uniquement la liste des personnages, rien d'autre, exclusivement <personnage1; personnage2>, etc :
"""

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Who won the world series in 2020?"},
  ]
)