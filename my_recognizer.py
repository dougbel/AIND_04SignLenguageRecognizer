import warnings
import math
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for test_item in range(0, test_set.num_items) :
        dictionary = {}
        for word, model in models.items():
            # calculate the scores for each model(word) and update the 'probabilities' list.
            __X, __length  = test_set.get_item_Xlengths(test_item)


            try:
                logl = model.score(__X, __length)
            except:
                logl = -math.inf
            dictionary[word] = logl

        probabilities.append(dictionary)
        # determine the maximum score for each model (word).
        maxloglword = max(dictionary, key=dictionary.get)
        guesses.append(maxloglword)


    return probabilities, guesses

