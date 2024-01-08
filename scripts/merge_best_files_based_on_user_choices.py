import csv
import json

def create_filtered_csv(file1, file2, json_file, output_file):
    # Charger les choix de l'utilisateur à partir du fichier JSON
    with open(json_file, 'r') as jfile:
        user_choices = json.load(jfile)

    # Ouvrir les fichiers CSV et le fichier de sortie
    with open(file1, newline='') as csvfile1, open(file2, newline='') as csvfile2, open(output_file, 'w', newline='') as output_csv:
        reader1 = csv.reader(csvfile1, delimiter=',', quotechar='"')
        reader2 = csv.reader(csvfile2, delimiter=',', quotechar='"')
        writer = csv.writer(output_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Lire les deux fichiers et écrire les lignes sélectionnées dans le fichier de sortie
        for row1, row2 in zip(reader1, reader2):
            chapter = row1[0]
            if chapter in user_choices:
                chosen_row = row1 if user_choices[chapter] == "1" else row2
                writer.writerow(chosen_row)

def main():
    file1 = "../submission_058.csv"  # Remplacez par le chemin d'accès au premier fichier CSV
    file2 = "../submission_070.csv"  # Remplacez par le chemin d'accès au second fichier CSV
    json_file = "user_choices.json"  # Chemin d'accès au fichier JSON des choix
    output_file = "filtered_output.csv"  # Nom du fichier CSV de sortie

    create_filtered_csv(file1, file2, json_file, output_file)

if __name__ == "__main__":
    main()
