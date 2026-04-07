#fasttext https://fasttext.cc/docs/en/python-module.html

#dimensions = 100

try:
    #!pip install fasttext
    import fasttext

    data = preprocessing("sample_data/train.tsv")

    model = fasttext.train_supervised(data)

    for word in model.words:
      pass
      print(word)
      print(model[word])
    print(model.labels)
    print(len(model.words))

except Exception as e :
    print(e)
