#HAndling argparse   
#accesed via <scriptname> <argument1> ..<argumentN>  eg: use_argparse.py abc def
#Makes the parser

import argparse



#accesed via <scriptname> <argument1> ..<argumentN>  eg: use_argparse.py abc def
try:
    #Makes the parser
    parser = argparse.ArgumentParser()
    #Add a couple of arguments
    parser.add_argument("dimensions", help="Number of dimensions.")  
    parser.add_argument("epochs", help="Number of epochs.")  
    parser.add_argument("inputfile", help="The input_file.")
    parser.add_argument("outputfile", help="The output_file.")
    #Parse the arguments given 
    args = parser.parse_args()
    #Use them
    dimensions = int(args.dimensions)
    output_file = args.outputfile
    input_file = args.inputfile
    epochs = int(args.epochs)
except Exception as e:
    dimensions = 100
    output_file = "out.bin"
    input_file = "train.tsv"
    epochs = 100
    print(e)



#replacing tab with space, treating latin charscters as a single word (including numbers)
# they apear to be surronded by space


#checks if the text contains latin characters
def contains_latin(text):
    latin="abcdefghijklmnopqrstuvwxyz1234567890"
    for char in text:
        if char.lower() in latin:
            return True
    return False


#splitting the text int separate "words", treating one chinese letter as a word and
def tokenize(text):
    result =""
    separated = text.split(" ")
    for word in separated:
        if contains_latin(word):
            result += " "+word
        else:
            for char in word:
                result += " "+char

    result = result.replace("  "," ")

    return result


#taking a list of strings as argument
def preprocessing(text):
    result = ""
    for line in text:
        try:
            label = "__label__"+line.split("\t")[1]
            sentence = tokenize(line.split("\t")[2])
            result += label+" "+sentence

        except:
            pass
    result = result.replace("  "," ")
    #print(result)
    return result


#fasttext https://fasttext.cc/docs/en/python-module.html


def run_fasttext():
    try:
        import fasttext

        file = open(input_file, "r")
        data_lines = file.readlines()[1:]
        file.close()

        data = preprocessing(data_lines)
        text_file = open("temp.txt", "w")
        text_file.write(data)
        text_file.close()

        #model = fasttext.train_supervised("temp.txt", dim = dimensions)
        model = fasttext.train_unsupervised(input="temp.txt", dim = dimensions, model='skipgram', epoch = epochs)
        #model = fasttext.train_unsupervised(input="temp.txt", dim = dimensions, model='cbow', epoch = epochs)

    except Exception as e :
        print(e)
    import os
    file_path = "temp.txt"
    try:
        os.remove(file_path)
    except Exception as e:
        print(e)

    model.save_model(output_file)


try:
    run_fasttext()
except Exception as e:
    print(e)
print("end")
