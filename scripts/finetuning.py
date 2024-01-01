import pandas as pd
from transformers import (
    CamembertForTokenClassification,
    AutoTokenizer,
)
import tqdm
import torch
import matplotlib.pyplot as plt


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        # Tokenize the input text
        tokenized = self.tokenizer(
            tokens,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized.input_ids.squeeze(0)

        # Apply padding of 0 in the labels array
        padded_labels = torch.nn.functional.pad(
            torch.tensor(labels), (0, self.max_length - len(labels)), value=0
        )

        return input_ids, padded_labels


def create_train_test_dataset(csv_file1, csv_file2, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")

    # Read csv files
    data1 = pd.read_csv(
        csv_file1, converters={"ner_tags": eval, "labels": eval}
    )
    data2 = pd.read_csv(
        csv_file2, converters={"ner_tags": eval, "labels": eval}
    )

    # Merge data from both files
    merged_data = (
        pd.concat([data1, data2]).sample(frac=1).reset_index(drop=True)
    )

    # Take only the values in the last 2 columns of the csv
    merged_data = merged_data.iloc[:, -2:]

    # Split data into train and test datasets
    train_data = merged_data.sample(frac=0.8, random_state=1)
    test_data = merged_data.drop(train_data.index)

    train_data = list(zip(train_data["tokens"], train_data["ner_tags"]))
    test_data = list(zip(test_data["tokens"], test_data["ner_tags"]))

    max_length = max(
        max(len(tokens) for tokens, _ in train_data),
        max(len(tokens) for tokens, _ in test_data),
    )

    train_dataset = CustomDataset(train_data, tokenizer, max_length)
    test_dataset = CustomDataset(test_data, tokenizer, max_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    return train_dataloader, test_dataloader


def save_model(model, path):
    model.save_pretrained(path)


def train_step(model, tokens, labels):
    labels = labels.argmax(-1)
    print(labels)
    loss = model(**tokens, labels=labels).loss

    return loss


def train(model, train_dataloader, epochs, optimizer):
    loss_array = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_dataloader):
            tokens, labels = batch
            optimizer.zero_grad()
            outputs = model(tokens, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataloader)}")
        loss_array.append(epoch_loss / len(train_dataloader))

    return loss_array


def plot_loss(loss_array):
    plt.plot(loss_array)
    plt.show()


def evaluate(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_count = 0
        for batch in tqdm.tqdm(test_dataloader):
            tokens, labels = batch
            outputs = model(tokens, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            total_correct += (predictions == labels).sum().item()
            total_count += len(labels[0])
        accuracy = total_correct / total_count
        print(f"Total correct: {total_correct}, Total count: {total_count}")
        print(
            f"Accuracy: {accuracy}, Loss: {total_loss / len(test_dataloader)}"
        )


if "__main__" == __name__:
    train_data, test_data = create_train_test_dataset(
        "data/finetuning_data/chapter_1.csv",
        "data/finetuning_data/chapter_2.csv",
        batch_size=16,
    )
    # print(len(train_data))
    # print(len(test_data))

    model = CamembertForTokenClassification.from_pretrained(
        "Jean-Baptiste/camembert-ner", return_dict=True
    )
    print("Evaluation before training")
    evaluate(model, test_data)
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    print("Training........")
    loss_array = train(model, train_data, 10, optimizer)
    plot_loss(loss_array)
    save_model(model, "models/finetuned_ner")
    print("Evaluation after training")
    evaluate(model, test_data)
