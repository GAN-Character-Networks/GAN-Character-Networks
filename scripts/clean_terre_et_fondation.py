"""
"""
import re


def process_file(file_path: str) -> None:
    """
    Function to apply the regex and replace the content of the file
    """
    try:
        with open(file_path, "r", errors="ignore") as file:
            content: str = file.read()

        # Part to remove artifacts in "TERRE ET FONDATION"
        content = re.sub(r"TERRE ET FONDATION", "", content)
        content = re.sub(r"\d{3}", "", content)

        with open(file_path, "w") as file:
            file.write(content)

        print("Le contenu du fichier a été modifié avec succès.")

    except FileNotFoundError:
        print(f"Le fichier '{file_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")


if __name__ == "__main__":
    process_file("data/txts/Terre_et_fondation_sample.txt")
