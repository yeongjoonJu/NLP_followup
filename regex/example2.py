import sys, fileinput, re
import nltk
from nltk.tokenize import sent_tokenize

if __name__=='__main__':
    # nltk.download('punkt')  
    for line in fileinput.input():
        if line.strip() != "":
            line = re.sub(r"([a-z])\.([a-z])", r'\1. \2', line.strip())
            print(line)
            sentences = sent_tokenize(line.strip())

            for s in sentences:
                if s != "":
                    sys.stdout.write(s + "\n")