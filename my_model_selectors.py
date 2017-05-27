import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        for numComponents in range(self.min_n_components, self.max_n_components + 1):
            print(numComponents)

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selecti BICon based on scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        maxScore = -math.inf
        maxModel = None

        for numComponents in range(self.min_n_components, self.max_n_components + 1):
            try:
                if self.verbose:
                    print("\n\n     WORKING FOR WORD {} PARA {} ESTADOS EN HMM".format(self.this_word, numComponents))
                    print("                      {} WITH {} SEQUENCES, NUMBER OF FOLDS CHOOSEN {}".format(self.this_word, len(self.sequences),min(3, len(self.sequences))))

                split_method = KFold(n_splits=min(3, len(self.sequences)))
                #restarting collection of scores
                scores = []
                numFold= 0

                # splitting in training and test sets
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    numFold += 1
                    if self.verbose:
                        print("     Fold number {} ".format(numFold))
                    #### TRAINING ####
                    # get fold for training
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    # train
                    model = self.base_model(numComponents)
                    ### restoring X and lenghts after training
                    self.X, self.lengths = self.hwords[self.this_word]

                    #### SCORING ####
                    # get fold for testing
                    __x, __length = combine_sequences(cv_test_idx, self.sequences)
                    # score
                    logl = model.score(__x, __length)
                    scores.append(logl)
                    if self.verbose:
                        print("           score {} ".format(logl))
                #getting mean of scores and model
                score, model = np.mean(scores), model
                if self.verbose:
                    print("     Average score {} ".format(logl))


                if score > maxScore:
                    maxScore = score
                    maxModel = model
                    if self.verbose:
                        print("     {} components with bigger score until now".format(numComponents))
            except:
                if self.verbose:
                    print("                      FAIL TRAINING FOR {} COMPONENTS IN HMM".format(numComponents))
                break
        return maxModel

