
# Part 4 - Evaluation

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import re
import argparse
from FFN.py import FFN

#accesed via <scriptname> <argument1> ..<argumentN>  eg: use_argparse.py abc def
try:
    #Makes the parser
    parser = argparse.ArgumentParser()
    #Add a couple of arguments 
    parser.add_argument("testfile", help="The test_file.")
    parser.add_argument("savefile", help="The output_file.")
    #Parse the arguments given 
    args = parser.parse_args()
    #Use them
    test_file = args.testfile #the test.tsv
    save_file_dir = args.savefile #where the model is saved
except Exception as e:
    test_file = "test.tsv"
    save_file_dir = "torch_output.bin"
    print(e)


model = torch.load(save_file_dir, weights_only=False)

# loading the model from earlier
#model = torch.load(save_file_dir, weights_only =True)
model.eval()

# getting data from test.tsv
texts = []
labels = []

with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    #for line in f:
    for i in range(1, len(lines), 1):
        line = lines[i].strip()
        label = line.split("\t")[1]
        text = line.split("\t")[2]
        texts.append(text)
        labels.append(label)

# getting the labels
unique_labels = sorted(list(set(labels)))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
idx_to_label = {i: l for l, i in label_to_idx.items()}

y_true = []
y_pred = []

# trying to predict results
with torch.no_grad():
    for text, label in zip(texts, labels):
        x = torch.tensor(sentence_vector(text), dtype=torch.float32)

        output = model(x)
        pred = torch.argmax(output).item()

        y_true.append(label_to_idx[label])
        y_pred.append(pred)

#  Accuracy
accuracy = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)

print(f"Accuracy: {accuracy:.4f}")

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# (valfritt: snyggare utskrift)
print("\nLabels:")
for i, l in idx_to_label.items():
    print(f"{i}: {l}")
    
    
print("end")
