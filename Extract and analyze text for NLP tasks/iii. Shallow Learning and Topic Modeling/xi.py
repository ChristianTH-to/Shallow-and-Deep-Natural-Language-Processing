'''References:  
   https://www.aclweb.org/anthology/D09-1026
   http://www.arbylon.net/publications/text-est.pdf
   http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
   Jiahong Zhou's L-LDA Implementation'''

# Stdlib
from concurrent import futures
import copy_reg
import types
import os
import json

# Third party
import numpy as np


# Implementation of labeled latent dirichlet allocation (L-LDA) using gibbs sampling
class LldaModel:

    def __init__(self, alpha_vector="50_div_K", eta_vector=None, labeled_documents=None):

        self.K = 0 # the number of topics
        self.M = 0 # the number of documents
        self.T = 0 # the number of terms
        self.WN = 0 # the number of all words in W
        self.LN = 0 # the number of all original labels
        self.W = [] # the corpus, a list of terms list, W[m] is the document vector, W[m][n] is the id of the term
        self.Z = [] # the topic corpus, just same as W, except Z[m][n] is the id of the topic of the term
        self.iteration = 0 # the number of iterations
        self.topics = [] # a list of the all topics
        self.terms = [] # a list of the all terms
        self.vocabulary = {} # a dict of <term, term_id>, vocabulary[terms[id]] == id
        self.topic_vocabulary = {} # a dict of <topic, topic_id>
        self.alpha_vector = alpha_vector # the prior distribution of theta_m, from parameter estimation from Gregor Heinrich
        self.eta_vector = eta_vector # the prior distribution of beta_k
        
        self.all_perplexities = []

        self.Lambda = None

        '''Derivative fields:
           The following fields could reduce operations in training and inference
           it is not necessary to save them to file, we can recover them by other fields.'''

        self.Doc2TopicCount = None # a matrix, shape is M * K, represents the times of topic k sampled in document m
        self.Topic2TermCount = None # a matrix, shape is K * T, represents the times of term t generated from topic k
        self.alpha_vector_Lambda = None # a matrix
        self.eta_vector_sum = 0.0 # float value and the prior distribution of beta_k
        self.Topic2TermCountSum = None # a vector

        # an iterable of a tuple, contains all doc and their labels
        if labeled_documents is not None:
         
            self._load_labeled_documents(labeled_documents)

        pass


    def _initialize_derivative_fields(self):
        """
        Initialize derivative fields
        """
        self.Doc2TopicCount = np.zeros((self.M, self.K), dtype=int)
        self.Topic2TermCount = np.zeros((self.K, self.T), dtype=int)
      
        for m in range(self.M):
            
            for t, z in zip(self.W[m], self.Z[m]):
               
                k = z
                self.Doc2TopicCount[m, k] += 1
                self.Topic2TermCount[k, t] += 1

        self.alpha_vector_Lambda = self.alpha_vector * self.Lambda
        self.eta_vector_sum = sum(self.eta_vector)
        self.Topic2TermCountSum = self.Topic2TermCount.sum(axis=1)

      
    def _load_labeled_documents(self, labeled_documents):
        """
        Input labeled corpus, which contains all documents and their corresponding labels
        labeled_documents: a iterable of tuple(doc, iterable of label), contains all doc and their labels
        """
        all_labels = []
        all_words = []
        doc_corpus = []
        labels_corpus = []
        for document, labels in labeled_documents:
            doc_words = document.split()
            doc_corpus.append(doc_words)
            if labels is None:
                labels = []
            labels.append("common_topic")
            labels_corpus.append(labels)
            all_words.extend(doc_words)
            all_labels.extend(labels)
        self.terms = list(set(all_words))
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}
        self.topics = list(set(all_labels))
        self.topic_vocabulary = {topic: index for index, topic in enumerate(self.topics)}
        self.K = len(self.topics)
        self.T = len(self.terms)
        self.W = [[self.vocabulary[term] for term in doc_words] for doc_words in doc_corpus]
        self.M = len(self.W)
        self.WN = len(all_words)

        '''We appended topic "common_topic" to each doc at the beginning
           so we need minus the number of "common_topic"
           LN is the number of original labels'''

        self.LN = len(all_labels) - self.M

        self.Lambda = np.zeros((self.M, self.K), dtype=float)
         
        for m in range(self.M):
         
            if len(labels_corpus[m]) == 1:
                labels_corpus[m] = self.topics
                  
            for label in labels_corpus[m]:
                k = self.topic_vocabulary[label]
                self.Lambda[m, k] = 1.0

        if self.alpha_vector is None:
         
            self.alpha_vector = [0.001 for _ in range(self.K)]
            
        elif type(self.alpha_vector) is str and self.alpha_vector == "50_div_K":
            self.alpha_vector = [50.0/self.K for _ in range(self.K)]
            
        elif type(self.alpha_vector) is float or type(self.alpha_vector) is int:
            self.alpha_vector = [self.alpha_vector for _ in range(self.K)]
            
        else:
            message = "error alpha_vector: %s" % self.alpha_vector
            raise Exception(message)

        if self.eta_vector is None:
         
            self.eta_vector = [0.001 for _ in range(self.T)]
            
        elif type(self.eta_vector) is float or type(self.eta_vector) is int:
            self.eta_vector = [self.eta_vector for _ in range(self.T)]
            
        else:
            message = "error eta_vector: %s" % self.eta_vector
            raise Exception(message)

        self.Z = []
      
        for m in range(self.M):
            
            numerator_vector = self.Lambda[m] * self.alpha_vector
            p_vector = 1.0 * numerator_vector / sum(numerator_vector)
            # z_vector is a vector of a document,
            # just like [2, 3, 6, 0], which means this doc have 4 word and them generated
            # from the 2nd, 3rd, 6th, 0th topic, respectively
            z_vector = [LldaModel._multinomial_sample(p_vector) for _ in range(len(self.W[m]))]
            self.Z.append(z_vector)

        self._initialize_derivative_fields()
      
        pass

      
    @staticmethod
    def _multinomial_sample(p_vector, random_state=None):
        """
        Sample a number from multinomial distribution
        p_vector: the probabilities
        returns a int value
        """
      
        if random_state is not None:
            
            return random_state.multinomial(1, p_vector).argmax()
        return np.random.multinomial(1, p_vector).argmax()

   
    def _gibbs_sample_training(self):
        """
        Sample a topic(k) for each word(t) of all documents, Generate a new matrix Z
        returns: None
        """
        count = 0
        for m in range(self.M):

            doc_m_alpha_vector = self.alpha_vector_Lambda[m]

            for t, z, n in zip(self.W[m], self.Z[m], range(len(self.W[m]))):
                k = z
                self.Doc2TopicCount[m, k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = self.Doc2TopicCount[m] + doc_m_alpha_vector

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]

                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                # theta_vector = 1.0 * numerator_theta_vector / denominator_theta
                # denominator_theta is independent with t and k, so denominator could be any value except 0
                # will set denominator_theta as 1.0
                theta_vector = numerator_theta_vector

                p_vector = beta_vector * theta_vector
                p_vector = 1.0 * p_vector / sum(p_vector)
                sample_z = LldaModel._multinomial_sample(p_vector)
                self.Z[m][n] = sample_z

                k = sample_z
                self.Doc2TopicCount[m, k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1
                count += 1
                  
        assert count == self.WN
        print "gibbs sample count: ", self.WN
        self.iteration += 1
        self.all_perplexities.append(self.perplexity)
         
        pass

   
    def _gibbs_sample_inference(self, term_vector, iteration=30):
        """
        Inference with gibbs sampling
        term_vector: the term vector of document
        iteration: the number of iterations
        returns: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """
        doc_topic_count = np.zeros(self.K, dtype=int)
        p_vector = np.ones(self.K, dtype=int)
        p_vector = p_vector * 1.0 / sum(p_vector)
        z_vector = [LldaModel._multinomial_sample(p_vector) for _ in term_vector]
      
        for n, t in enumerate(term_vector):
            
            k = z_vector[n]
            doc_topic_count[k] += 1
            self.Topic2TermCount[k, t] += 1
            self.Topic2TermCountSum[k] += 1

        doc_m_alpha_vector = self.alpha_vector

        for i in range(iteration):
            
            for n, t in enumerate(term_vector):
               
                k = z_vector[n]
                doc_topic_count[k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = doc_topic_count + doc_m_alpha_vector

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]

                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                # theta_vector = 1.0 numerator_theta_vector / denominator_theta
                # denominator_theta is independent with t and k, so denominator could be any value except 0
                # will set denominator_theta as 1.0
                theta_vector = numerator_theta_vector

                p_vector = beta_vector * theta_vector
                p_vector = 1.0 * p_vector / sum(p_vector)
                sample_z = LldaModel._multinomial_sample(p_vector)
                z_vector[n] = sample_z

                k = sample_z
                doc_topic_count[k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1

        for n, t in enumerate(term_vector):
         
            k = z_vector[n]
            self.Topic2TermCount[k, t] -= 1
            self.Topic2TermCountSum[k] -= 1

        numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
        denominator_theta = sum(numerator_theta_vector)
        theta_new = 1.0 * numerator_theta_vector / denominator_theta
      
        return theta_new
      

    def _gibbs_sample_inference_multi_processors(self, term_vector, iteration=30):
        """
        Inference with gibbs sampling
        term_vector: the term vector of document
        iteration: the times of iteration
        returns: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """
        # print("gibbs sample inference iteration: %s" % iteration)
        random_state = np.random.RandomState()
        topic2term_count = self.Topic2TermCount.copy()
        topic2term_count_sum = self.Topic2TermCountSum.copy()

        doc_topic_count = np.zeros(self.K, dtype=int)
        p_vector = np.ones(self.K, dtype=int)
        p_vector = p_vector * 1.0 / sum(p_vector)
        z_vector = [LldaModel._multinomial_sample(p_vector, random_state=random_state) for _ in term_vector]
         
        for n, t in enumerate(term_vector):
         
            k = z_vector[n]
            doc_topic_count[k] += 1
            topic2term_count[k, t] += 1
            topic2term_count_sum[k] += 1

        doc_m_alpha_vector = self.alpha_vector

        for i in range(iteration):
            
            for n, t in enumerate(term_vector):
               
                k = z_vector[n]
                doc_topic_count[k] -= 1
                topic2term_count[k, t] -= 1
                topic2term_count_sum[k] -= 1

                numerator_theta_vector = doc_topic_count + doc_m_alpha_vector

                numerator_beta_vector = topic2term_count[:, t] + self.eta_vector[t]

                denominator_beta = topic2term_count_sum + self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                # theta_vector = 1.0 numerator_theta_vector / denominator_theta
                # Denominator_theta is independent with t and k, so denominator could be any value except 0
                # will set denominator_theta as 1.0
                theta_vector = numerator_theta_vector

                p_vector = beta_vector * theta_vector

                p_vector = 1.0 * p_vector / sum(p_vector)

                sample_z = LldaModel._multinomial_sample(p_vector, random_state)
                z_vector[n] = sample_z

                k = sample_z
                doc_topic_count[k] += 1
                topic2term_count[k, t] += 1
                topic2term_count_sum[k] += 1


        numerator_theta_vector = doc_topic_count + doc_m_alpha_vector

        denominator_theta = sum(numerator_theta_vector)
        theta_new = 1.0 * numerator_theta_vector / denominator_theta
        return theta_new

    def training(self, iteration=10, log=False):
        """
        Training this model with gibbs sampling
        log: print perplexity after every gibbs sampling if True
        iteration: the times of iteration
        returns: None
        """
        for i in range(iteration):
         
            if log:
                print "after iteration: %s, perplexity: %s" % (self.iteration, self.perplexity)
            self._gibbs_sample_training()
            
        pass

   
    def inference(self, document, iteration=30, times=10):
    
        """
        Inference for one document
        times: the times of gibbs sampling, the result is the average value of all times(gibbs sampling)
        iteration: the times of iteration
        document: some sentence like "this is a method for inference"
        returns: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """

        words = document.split()
        term_vector = [self.vocabulary[word] for word in words if word in self.vocabulary]
        theta_new_accumulation = np.zeros(self.K, float)
        for time in range(times):
            theta_new = self._gibbs_sample_inference(term_vector, iteration=iteration)
        
            theta_new_accumulation += theta_new
        theta_new = 1.0 * theta_new_accumulation / times
        
        doc_topic_new = [(self.topics[k], probability) for k, probability in enumerate(theta_new)]
        sorted_doc_topic_new = sorted(doc_topic_new,
                                      key=lambda topic_probability: topic_probability[1],
                                      reverse=True)
        return sorted_doc_topic_new
   
        pass

   
    def inference_multi_processors(self, document, iteration=30, times=8, max_workers=8):
        
        """
        Inference for one document
        times: the times of gibbs sampling, the result is the average value of all times(gibbs sampling)
        iteration: the times of iteration
        document: some sentence like "this is a method for inference"
        max_workers: the max number of processors(workers)
        returns: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """


        def _pickle_method(m):
         
            if m.im_self is None:
               
                return getattr, (m.im_class, m.im_func.func_name)
               
            else:
                return getattr, (m.im_self, m.im_func.func_name)
               
        copy_reg.pickle(types.MethodType, _pickle_method)

        words = document.split()
        term_vector = [self.vocabulary[word] for word in words if word in self.vocabulary]
        term_vectors = [term_vector for _ in range(times)]
        iterations = [iteration for _ in range(times)]

        with futures.ProcessPoolExecutor(max_workers) as executor:
            
            res = executor.map(self._gibbs_sample_inference_multi_processors, term_vectors, iterations)
        theta_new_accumulation = np.zeros(self.K, float)
      
        for theta_new in res:
            theta_new_accumulation += theta_new
        theta_new = 1.0 * theta_new_accumulation / times
        
        doc_topic_new = [(self.topics[k], probability) for k, probability in enumerate(theta_new)]
        sorted_doc_topic_new = sorted(doc_topic_new,
                                      key=lambda topic_probability: topic_probability[1],
                                      reverse=True)
        return sorted_doc_topic_new
   
        pass

   
    def beta_k(self, k):
        """
        topic-term distribution
        beta_k[t] is the probability of term t(word) to be generated from topic k
        :return: a vector, shape is T
        """
        numerator_vector = self.Topic2TermCount[k] + self.eta_vector
        
        denominator = sum(numerator_vector)
         
        return 1.0 * numerator_vector / denominator

   
    def theta_m(self, m):
        """
        doc-topic distribution
        theta_m[k] is the probability of doc m to be generated from topic k
        :return: a vector, shape is K
        """
        numerator_vector = self.Doc2TopicCount[m] + self.alpha_vector * self.Lambda[m]
        
        denominator = sum(numerator_vector)
         
        return 1.0 * numerator_vector / denominator

   
    @property
    def beta(self):
        """
        topic-term distribution
        beta[k, t] is the probability of term t(word) to be generated from topic k
        :return: a matrix, shape is K * T
        """
        numerator_matrix = self.Topic2TermCount + self.eta_vector
    
        # column vector
    
        denominator_vector = numerator_matrix.sum(axis=1).reshape(self.K, 1)
      
        return 1.0 * numerator_matrix / denominator_vector

        pass

      
    @property
    def theta(self):
        """
        doc-topic distribution
        theta[m, k] is the probability of doc m to be generated from topic k
        :return: a matrix, shape is M * K
        """
        numerator_matrix = self.Doc2TopicCount + self.alpha_vector * self.Lambda
        denominator_vector = numerator_matrix.sum(axis=1).reshape(self.M, 1)

        # column vector

        return 1.0 * numerator_matrix / denominator_vector
   
        pass

   
    @property
    def log_perplexity(self):
        """
        log perplexity of LDA topic model
        :return: a float value
        """
        beta = self.beta
        
        log_likelihood = 0
        word_count = 0
         
        for m, theta_m in enumerate(self.theta):
         
            for t in self.W[m]:
               
                likelihood_t = np.inner(beta[:, t], theta_m)
                
                log_likelihood += -np.log(likelihood_t)
                word_count += 1
               
        assert word_count == self.WN, "word_count: %s\tself.WN: %s" % (word_count, self.WN)
      
        return 1.0 * log_likelihood / self.WN

      
    @property
    def perplexity(self):
        """
        perplexity of LDA topic model
        :return: a float value, perplexity = exp{log_perplexity}
        """
      
        return np.exp(self.log_perplexity)

      
    def __repr__(self):
      
        return "\nLabeled-LDA Model:\n" \
               "\tK = %s\n" \
               "\tM = %s\n" \
               "\tT = %s\n" \
               "\tWN = %s\n" \
               "\tLN = %s\n" \
               "\talpha = %s\n" \
               "\teta = %s\n" \
               "\tperplexity = %s\n" \
               "\t" % (self.K, self.M, self.T, self.WN, self.LN, self.alpha_vector[0], self.eta_vector[0], self.perplexity)
      
        pass

      
    class SaveModel:
        def __init__(self, save_model_dict=None):
            self.alpha_vector = []
            self.eta_vector = []
            self.terms = []
            self.vocabulary = {}
            self.topics = []
            self.topic_vocabulary = {}
            self.W = []
            self.Z = []
            self.K = 0
            self.M = 0
            self.T = 0
            self.WN = 0
            self.LN = 0
            self.iteration = 0

            # The following fields cannot be dumped into json file
            # we need write them with np.save() and read them with np.load()
            self.Lambda = None

            if save_model_dict is not None:
               
                self.__dict__ = save_model_dict
                  
        pass

   
    @staticmethod
    def _read_object_from_file(file_name):
        """
        read an object from json file
        file_name: json file name
        returns: None if file doesn't exist or can not convert to an object by json, else return the object
        """
        if os.path.exists(file_name) is False:
         
            print ("Error read path: [%s]" % file_name)
            return None
         
        with open(file_name, 'r') as f:
            try:
                obj = json.load(f)
            except Exception:
                print ("Error json: [%s]" % f.read()[0:10])
                  
                return None
        return obj

   
    @staticmethod
    def _write_object_to_file(file_name, target_object):
        """
        Write the object to file with json(if the file exists, this function will overwrite it)
        file_name: the name of new file
        target_object: the target object for writing
        returns: True if success else False
        """
        dirname = os.path.dirname(file_name)
        LldaModel._find_and_create_dirs(dirname)
        try:
            with open(file_name, "w") as f:
                json.dump(target_object, f, skipkeys=False, ensure_ascii=False, check_circular=True, allow_nan=True, cls=None, indent=True, separators=None, encoding="utf-8", default=None, sort_keys=False)
        except Exception, e:
         
            message = "Write [%s...] to file [%s] error: json.dump error" % (str(target_object)[0:10], file_name)
            print ("%s\n\t%s" % (message, e.message))
            print "e.message: ", e.message
            
            return False
        else:
            return True

         
    @staticmethod
    def _find_and_create_dirs(dir_name):
        """
        Find dir, create it if it doesn't exist
        :param dir_name: the name of dir
        :return: the name of dir
        """
        if os.path.exists(dir_name) is False:
            
            os.makedirs(dir_name)
            
        return dir_name

   
    def save_model_to_dir(self, dir_name, save_derivative_properties=False):
        """
        save model to directory dir_name
        save_derivative_properties: save derivative properties if True
            some properties are not necessary save to disk, they could be derived from some basic properties,
            we call they derivative properties.
            To save derivative properties to disk:
                It will reduce the time of loading model from disk (read properties directly but do not compute them)
                but, meanwhile, it will take up more disk space
        dir_name: the target directory name
        :return: None
        """
        save_model = LldaModel.SaveModel()
        save_model.alpha_vector = self.alpha_vector
        save_model.eta_vector = self.eta_vector
        save_model.terms = self.terms
        save_model.vocabulary = self.vocabulary
        save_model.topics = self.topics
        save_model.topic_vocabulary = self.topic_vocabulary
        save_model.W = self.W
        save_model.Z = self.Z
        save_model.K = self.K
        save_model.M = self.M
        save_model.T = self.T
        save_model.WN = self.WN
        save_model.LN = self.LN
        save_model.iteration = self.iteration

        save_model_path = os.path.join(dir_name, "llda_model.json")
        LldaModel._write_object_to_file(save_model_path, save_model.__dict__)

        np.save(os.path.join(dir_name, "Lambda.npy"), self.Lambda)

        # Save derivative properties

        if save_derivative_properties:
         
            np.save(os.path.join(dir_name, "Doc2TopicCount.npy"), self.Doc2TopicCount)
            np.save(os.path.join(dir_name, "Topic2TermCount.npy"), self.Topic2TermCount)
            np.save(os.path.join(dir_name, "alpha_vector_Lambda.npy"), self.alpha_vector_Lambda)
            np.save(os.path.join(dir_name, "eta_vector_sum.npy"), self.eta_vector_sum)
            np.save(os.path.join(dir_name, "Topic2TermCountSum.npy"), self.Topic2TermCountSum)
            
        pass

   
    def load_model_from_dir(self, dir_name, load_derivative_properties=True):
        """
        Load model from directory dir_name
        load_derivative_properties: load derivative properties from disk if True
        dir_name: the target directory name
        returns: None
        """
        save_model_path = os.path.join(dir_name, "llda_model.json")
        save_model_dict = LldaModel._read_object_from_file(save_model_path)
        save_model = LldaModel.SaveModel(save_model_dict=save_model_dict)
        self.alpha_vector = save_model.alpha_vector
        self.eta_vector = save_model.eta_vector
        self.terms = save_model.terms
        self.vocabulary = save_model.vocabulary
        self.topics = save_model.topics
        self.topic_vocabulary = save_model.topic_vocabulary
        self.W = save_model.W
        self.Z = save_model.Z
        self.K = save_model.K
        self.M = save_model.M
        self.T = save_model.T
        self.WN = save_model.WN
        self.LN = save_model.LN
        self.iteration = save_model.iteration

        self.Lambda = np.load(os.path.join(dir_name, "Lambda.npy"))

        # Load load_derivative properties
        if load_derivative_properties:
         
            try:
                self.Doc2TopicCount = np.load(os.path.join(dir_name, "Doc2TopicCount.npy"))
                self.Topic2TermCount = np.load(os.path.join(dir_name, "Topic2TermCount.npy"))
                self.alpha_vector_Lambda = np.load(os.path.join(dir_name, "alpha_vector_Lambda.npy"))
                self.eta_vector_sum = np.load(os.path.join(dir_name, "eta_vector_sum.npy"))
                self.Topic2TermCountSum = np.load(os.path.join(dir_name, "Topic2TermCountSum.npy"))
            except IOError or ValueError, e:
                print("%s: load derivative properties fail, initialize them with basic properties" % e)
                self._initialize_derivative_fields()
        else:
            self._initialize_derivative_fields()
            
        pass

   
    def update(self, labeled_documents=None):
        """
        Update model with labeled documents, incremental update
        :return: None
        """
        self.all_perplexities = []
        if labeled_documents is None:
         
            pass

        new_labels = []
        new_words = []
        new_doc_corpus = []
        new_labels_corpus = []
        for document, labels in labeled_documents:
            doc_words = document.split()
            new_doc_corpus.append(doc_words)
            if labels is None:
                labels = []
            labels.append("common_topic")
            new_labels_corpus.append(labels)
            new_words.extend(doc_words)
            new_labels.extend(labels)
        
        new_terms = set(new_words) - set(self.terms)
        self.terms.extend(new_terms)
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}

        
        new_topics = set(new_labels) - set(self.topics)
        self.topics.extend(new_topics)
        self.topic_vocabulary = {topic: index for index, topic in enumerate(self.topics)}

        old_K = self.K
        old_T = self.T
        self.K = len(self.topics)
        self.T = len(self.terms)

        
        new_w_vectors = [[self.vocabulary[term] for term in doc_words] for doc_words in new_doc_corpus]
         
        for new_w_vector in new_w_vectors:
         
            self.W.append(new_w_vector)

        old_M = self.M
        old_WN = self.WN
        self.M = len(self.W)
        self.WN += len(new_words)

        # We appended topic "common_topic" to each doc at the beginning
        # so we need minus the number of "common_topic"
        # LN is the number of original labels

        old_LN = self.LN

        self.LN += len(new_labels) + len(new_labels_corpus)

        old_Lambda = self.Lambda
        self.Lambda = np.zeros((self.M, self.K), dtype=float)
      
        for m in range(self.M):
            
            if m < old_M:

                # If the old document has no topic, we also init it to all topics here

                if sum(old_Lambda[m]) == old_K:
                    
                    self.Lambda[m] += 1.0
                continue
            
            if len(new_labels_corpus[m-old_M]) == 1:
                new_labels_corpus[m-old_M] = self.topics
            for label in new_labels_corpus[m-old_M]:
                k = self.topic_vocabulary[label]
                self.Lambda[m, k] = 1.0

        # The following 2 fields should be modified again if alpha_vector is not constant vector
        self.alpha_vector = [self.alpha_vector[0] for _ in range(self.K)]
        self.eta_vector = [self.eta_vector[0] for _ in range(self.T)]

        
        for m in range(old_M, self.M):
            
            numerator_vector = self.Lambda[m] * self.alpha_vector
            p_vector = numerator_vector / sum(numerator_vector)

            # z_vector is a vector of a document,
            # just like [2, 3, 6, 0], which means this doc have 4 word and them generated
            # from the 2nd, 3rd, 6th, 0th topic, respectively
            z_vector = [LldaModel._multinomial_sample(p_vector) for _ in range(len(self.W[m]))]
            self.Z.append(z_vector)

        self._initialize_derivative_fields()
      
        pass

      
    @staticmethod
    def _extend_matrix(origin=None, shape=None, padding_value=0):
        """
        For quickly extend the matrices when update
        extend origin matrix with shape, padding with padding_value
        :type shape: the shape of new matrix
        origin: np.ndarray, the original matrix
        returns: np.ndarray, a matrix with new shape
        """
        new_matrix = np.zeros(shape, dtype=origin.dtype)

        for row in range(new_matrix.shape[0]):
         
            for col in range(new_matrix.shape[1]):
               
                if row < origin.shape[0] and col < origin.shape[0]:
                     
                    new_matrix[row, col] = origin[row, col]
                else:
                    new_matrix[row, col] = padding_value

        return new_matrix
   
        pass

   
    @property
    def is_convergent(self):
        """
        Is this model convergent?
        :return: True if model is convergent
        """
        if len(self.all_perplexities) < 10:
         
            return False
         
        perplexities = self.all_perplexities[-10:]
      
        if max(perplexities) - min(perplexities) > 0.5:
            
            return False
        return True


if __name__ == "__main__":
    pass
