
#Pytorch part
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import fasttext
import re
import argparse



#accesed via <scriptname> <argument1> ..<argumentN>  eg: use_argparse.py abc def
try:
    #Makes the parser
    parser = argparse.ArgumentParser()
    #Add a couple of arguments
    parser.add_argument("epochs", help="Number of epochs.")  
    parser.add_argument("inputfile", help="The input_file.")
    parser.add_argument("outputfile", help="The output_file.")
    parser.add_argument("savefile", help="The output_file.")
    parser.add_argument("batchsize", help="The output_file.")
    parser.add_argument("learningrate", help="The output_file.")
    #Parse the arguments given 
    args = parser.parse_args()
    #Use them
    output_file = args.outputfile #the model from fasttext
    input_file = args.inputfile #the train.tsv
    epochs = int(args.epochs)
    save_file_dir = args.savefile #where the model is saved
    batch_size = int(args.batchsize)
    learning_rate = float(args.learningrate)
except Exception as e:
    output_file = "out.bin"
    input_file = "train.tsv"
    epochs = 100
    save_file_dir = "torch_output.bin"
    batch_size = 100
    learning_rate = 0.01
    print(e)










#read the model from the bin-file
fastxt_mod = fasttext.load_model(output_file)

EMBED_DIM = fastxt_mod.get_dimension()

#read the tsv
def load_data(path):
    texts = []
    labels = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            if len(line.split("\t"))>1:
              line = line.strip()
              label = line.split("\t")[1]
              text = line.split("\t")[2]
              texts.append(text)
              labels.append(label)

    return texts, labels


def encode_labels(labels):
    unique = list(set(labels))
    label2idx = {l: i for i, l in enumerate(unique)}
    idx2label = {i: l for l, i in label2idx.items()}

    encoded = [label2idx[l] for l in labels]
    return encoded, label2idx, idx2label




def sentence_vector(text):
    # lower and takes away ., etc
    words = re.findall(r'\b\w+\b', text.lower())

    vecs = [fastxt_mod.get_word_vector(w) for w in words]

    if len(vecs) == 0:
        return np.zeros(EMBED_DIM)

    vec = np.mean(vecs, axis=0)
    return vec / (np.linalg.norm(vec) + 1e-8)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.X = [sentence_vector(t) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class FFN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            ## So here is the pattern of layers
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# training it
def train(model, dataloader, epochs=epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        tot_loss = 0

        for X, y in dataloader:
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss : {tot_loss:.4f}")





texts, labels = load_data(input_file)
labels_encoded, label2idx, idx2label = encode_labels(labels)

dataset = TextDataset(texts, labels_encoded)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = FFN(EMBED_DIM, len(label2idx))

train(model, dataloader, epochs=epochs)

# saving the model to a bin file. NAme of file is based on user input
#torch.save(model.state_dict(), save_file_dir)
torch.save(model, save_file_dir)
print("end")