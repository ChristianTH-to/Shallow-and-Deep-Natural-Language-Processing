![thumbnail_original-1](https://user-images.githubusercontent.com/29679899/59774825-207b5900-927e-11e9-8560-f8c8c454ec25.png)
![thumbnail_original](https://user-images.githubusercontent.com/29679899/59774888-3852dd00-927e-11e9-812d-61dc8d47af1a.png)

Extracted over 300k text documents from Facebook Messenger 
users occupying the same group for 7 years. 

Cleaned, prepared and transformed semi-structured data into a dataset. 

Performed exploratory analysis using dimensionality reduction techniques PCA and t-SNE. 

Used multiple frequentist, Bayesian and deep learning classification algorithms to identify 
unlabeled text weighted by Word2vec and GloVe word embeddings. 

Derived topics and sentiment from unstructured data using semi-supervised and 
unsupervised latent dirichlet allocation. 

Preserved a global view of user topics and distilled each user into a mixture of topics using t-SNE.


i. Extraction and Exploration

	i. Extraction, exploration, feature extraction
	ii. Zipf’s Law
	iii. Initial cleaning and common-word visualization
	iv. Feature distributions by user and more cleaning

ii. Deep Learning

	v. Additional exploration with Word2vec and GloVe
	vi. Softmax, LeakyReLU and Sigmoid visualizations 
	vii. Export Word2vec embeddings
	viii. Diagnosing text using edm
	x. Attention based bidirectional lstm rnn

iii. Shallow learning and topic modeling

	xi. tfidf-cv weighted embeddings trained with: 

		i. logistic regression 
		ii. naive bayes 
		iii. xgboost 
		iv. support vector machines

	xii. Topic modeling with vanilla-lda
	xiii. L-lda implementation with inference via Gibbs sampling
	xiv. Defined l-lda seeds
	xv. Compare distance of user probability distributions
  
  
 You can find the original blog post <a href="https://www.xtiandata.com/single-post/2018/10/26/Shallow-Deep-Natural-Language-Processing" rel="nofollow">here</a></li>.
