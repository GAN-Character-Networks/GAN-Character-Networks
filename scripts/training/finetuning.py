import pandas as pd
from transformers import (
    CamembertForTokenClassification,
    AutoTokenizer,
)
import tqdm
import torch
import matplotlib.pyplot as plt
import ast
from torch.nn.utils.rnn import pad_sequence

def seed_everything(seed: int) -> None:
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        
        indices = self.tokenizer.convert_tokens_to_ids([self.bos_token] + tokens + [self.eos_token])
        indices = torch.tensor(indices, dtype=torch.long)
        labels = [0] + labels + [0]

        return indices, torch.tensor(labels, dtype=torch.long)

def load_data_from_csv(csv_file):
    return pd.read_csv(csv_file, index_col='id', encoding='utf-8')

def collate_fn(batch):
    # Unzip the batch into separate lists for inputs and labels
    inputs, labels = zip(*batch)

    # tokenizer.pad_token = 1
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=1)
    attention_mask = torch.where(padded_inputs != 1, torch.tensor(1), torch.tensor(0))
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_inputs, attention_mask, padded_labels

def create_train_test_dataset(csv_file, tokenizer, batch_size=32):
    # Load data from CSV files
    datasets = load_data_from_csv(csv_file)
    
    # Split data into train and test datasets
    train_data = datasets.sample(frac=0.8, random_state=1)
    test_data = datasets.drop(train_data.index)
    valid_data = test_data.sample(frac=0.5, random_state=1)
    test_data = test_data.drop(valid_data.index)

    train_data = list(zip(train_data["tokens"], train_data["ner_tags"]))
    test_data = list(zip(test_data["tokens"], test_data["ner_tags"]))
    valid_data = list(zip(valid_data["tokens"], valid_data["ner_tags"]))

    # conver the string representation of the list to a list
    train_data = [(ast.literal_eval(tokens), ast.literal_eval(labels)) for tokens, labels in train_data]
    test_data = [(ast.literal_eval(tokens), ast.literal_eval(labels)) for tokens, labels in test_data]
    valid_data = [(ast.literal_eval(tokens), ast.literal_eval(labels)) for tokens, labels in valid_data]

    train_dataset = CustomDataset(train_data, tokenizer)
    test_dataset = CustomDataset(test_data, tokenizer)
    valid_dataset = CustomDataset(valid_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader

def train(model, valid_dataloader, train_dataloader, epochs, optimizer, device):
    loss_array = []
    best_accuracy = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_dataloader):
            tokens, attention_mask, labels = batch
        
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(tokens, labels=labels, attention_mask=attention_mask)
        
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataloader)}")
        acc = evaluate(model, valid_dataloader, device)
        if acc > best_accuracy:
            best_accuracy = acc
            # save the model
            print("Saving the model at epoch ", epoch)
            torch.save(model.state_dict(), "best_model.pt")
        loss_array.append(epoch_loss / len(train_dataloader))

    return loss_array

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for batch in tqdm.tqdm(dataloader):
        tokens, attention_mask, labels = batch
    
        tokens = tokens.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(tokens, labels=labels, attention_mask=attention_mask)
    
        total_loss += outputs.loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        total_correct += torch.sum((predictions == labels) & (labels != 0))
        total_count += torch.sum(labels != 0)
    
    accuracy = (total_correct / total_count) * 100
    print(f"Total correct: {total_correct}, Total count: {total_count}")
    print(
        f"Accuracy: {accuracy}, Loss: {total_loss / len(test_dataloader)}"
    )
    return accuracy


if "__main__" == __name__:
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = "Jean-Baptiste/camembert-ner"
    seed = 42

    seed_everything(seed)

    model = CamembertForTokenClassification.from_pretrained(source, return_dict=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(source)

    train_dataloader, valid_dataloader, test_dataloader = create_train_test_dataset(
        "data/finetuning_data/train_data.csv",
        tokenizer,
        batch_size=batch_size,
    )

    print("training dataset length: ", len(train_dataloader))
    print("validation dataset length: ", len(valid_dataloader))
    print("test dataset length: ", len(test_dataloader))
    
    print("Zero shot evaluation on validation dataset")
    evaluate(model, valid_dataloader, device)

    print("Zero shot evaluation on test dataset")
    evaluate(model, test_dataloader, device)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print("Training........")
    loss_array = train(model, valid_dataloader, train_dataloader, 10, optimizer, device)

    # load the best model
    print("Loading the best model")
    model.load_state_dict(torch.load("best_model.pt"))

    print("Evaluation after training")
    evaluate(model, test_dataloader, device)
