r"""This script allows to merge two CSV files based on the user choices.

Authors
-------
 * Gabriel DESBOUIS 2023
"""


import csv
import json

def create_filtered_csv(file1, file2, json_file, output_file):
    with open(json_file, 'r') as jfile:
        user_choices = json.load(jfile)

    with open(file1, newline='') as csvfile1, open(file2, newline='') as csvfile2, open(output_file, 'w', newline='') as output_csv:
        reader1 = csv.reader(csvfile1, delimiter=',', quotechar='"')
        reader2 = csv.reader(csvfile2, delimiter=',', quotechar='"')
        writer = csv.writer(output_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row1, row2 in zip(reader1, reader2):
            chapter = row1[0]
            if chapter in user_choices:
                chosen_row = row1 if user_choices[chapter] == "1" else row2
                writer.writerow(chosen_row)

def main():
    file1 = "../submission_058.csv"  # Remplace by the path to the first CSV file
    file2 = "../submission_070.csv"  # Remplace by the path to the second CSV file
    json_file = "user_choices.json"  # Filename of the JSON file containing the user choices
    output_file = "filtered_output.csv"  # Name of the output CSV file

    create_filtered_csv(file1, file2, json_file, output_file)

if __name__ == "__main__":
    main()
