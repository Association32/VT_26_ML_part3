Requirements are

fasttext

argparse

torch

torch.nn as nn

torch.optim as optim

numpy as np

fasttext

numpy

sklearn 

re







The project felt a bit like a black box

The model achieved an accuracy of 0.16.

Given 6 classes, random chance corresponds to an accuracy of approximately 0.17.  
Thus, the model performs at or near chance level.

Although the training loss decreased significantly, the model does not generalize well to the test set.

Possible reasons might be that the train and test set are not matching very well. I speak no chinese, so I just have to trust these things
It is also possible that the feed-forward network simply is not good enough to handle this



The confusion matrix indicates that the model has problems to distinguish between most classes and tends to predict a small subset of labels more frequently.
This points to that the learned representations are not very distinct.

