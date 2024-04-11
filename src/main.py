#!/usr/bin/env python3
from tagger import Tagger
import sys

def main():
    tagger = Tagger()
    tagger.train(sys.argv[1]) # Train the model
    tagger.save('tagger.model') # Save it
    print(tagger.eval('tagger-eval.tsv')) # Evaluate it on some testing data 
    
if __name__ == "__main__":
    main()