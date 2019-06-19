# Export word2vec embeddings

# Data

import pandas as pd
dataset = pd.read_csv('bt_train_data.csv').fillna('')

# Dependencies

import re
#import nltk
import gensim
import numpy as np
#extra_stopwords = []
#from nltk.corpus import stopwords
#from stop_words import get_stop_words
#from nltk.stem.porter import PorterStemmer 
#from gensim.parsing.preprocessing import STOPWORDS


# Lowercases and splits text into tokens

corpus = []
for i in range(0, 302731):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(dataset['Message'][i]))
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    #clean_text = [word for word in clean_text if not word in set(stopwords.words('english'))]
    #clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
    #clean_text = [word for word in clean_text if word not in STOPWORDS]
    #clean_text = [word for word in clean_text if word not in extra_stopwords]
    # text stemming
    #ps = PorterStemmer() 
    #clean_text = [ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
    #clean_text = ' '.join(clean_text) # comment out to tokenize & create metadata <- not accurate, as I need all brackets removed and keeping this function retains bracketless information
    #clean_text = np.asarray(clean_text) # put list into array (2D? 3D?)
    #clean_text = clean_text.split()
    #clean_text = re.sub('\s+','\n', clean_text, re.M | re.DOTALL)
    corpus.append(clean_text)  


# Word2vec model

from gensim.models import Word2Vec
from gensim import models
bp2Vec = Word2Vec(sentences = corpus, size = 300, window = 5, min_count = 4, workers = 8,
                  sg = 0, iter = 30, alpha = 0.020)


# saves Word2vec as a .csv preserving the embedding dimensions

bp2Vec.wv.save_word2vec_format('bt2vec_embeddings_nostopwords_300d.csv', binary = False)


# Transforming word2vec embeddings into tensors for t-SNE implementation

'''Source code that I used to convert the word2vec embeddings into a 2D tensor and metadata 
format so that they could be used with Tensorflows embedding projector.''' 

# Copyright (C) 2016 Loreto Parisi <loretoparisi@gmail.com>
# Copyright (C) 2016 Silvio Olivastri <silvio.olivastri@gmail.com>
# Copyright (C) 2016 Radim Rehurek <radim@rare-technologies.com>

'''Command line arguments
----------------------

.. program-output:: python -m gensim.scripts.word2vec2tensor --help
   :ellipsis: 0, -7'''

# Dependencies

import os
import sys
import logging
import argparse
from smart_open import smart_open
import gensim

logger = logging.getLogger(__name__)

def word2vec2tensor(word2vec_model_path, tensor_filename, binary=False):
    """Convert file in Word2Vec format and writes two files 2D tensor TSV file.
    File "tensor_filename"_tensor.tsv contains word-vectors, "tensor_filename"_metadata.tsv contains words.
    Parameters
    ----------
    word2vec_model_path : str
        Path to file in Word2Vec format.
    tensor_filename : str
        Prefix for output files.
    binary : bool, optional
        True if input file in binary format.
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'

    with smart_open(outfiletsv, 'wb') as file_vector, smart_open(outfiletsvmeta, 'wb') as file_metadata:
        for word in model.index2word:
            file_metadata.write(gensim.utils.to_utf8(word) + gensim.utils.to_utf8('\n'))
            vector_row = '\t'.join(str(x) for x in model[word])
            file_vector.write(gensim.utils.to_utf8(vector_row) + gensim.utils.to_utf8('\n'))

    logger.info("2D tensor file saved to %s", outfiletsv)
    logger.info("Tensor metadata file saved to %s", outfiletsvmeta)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__[:-138])
    parser.add_argument("-i", "--input", required=True, help="Path to input file in word2vec format")
    parser.add_argument("-o", "--output", required=True, help="Prefix path for output files")
    parser.add_argument(
        "-b", "--binary", action='store_const', const=True, default=False,
        help="Set this flag if word2vec model in binary format (default: %(default)s)"
    )
    args = parser.parse_args()

    logger.info("running %s", ' '.join(sys.argv))
    word2vec2tensor(args.input, args.output, args.binary)
    logger.info("finished running %s", os.path.basename(sys.argv[0]))

    # Run the source code above in the terminal with these commands
    
    # $ python3 -m gensim.scripts.word2vec2tensor -i bt_4_2vec.txt -o Desktop
