# Esse notebook ilustra como fazer uma analise de sentimento usando dados do twitter e o pacote NLTK do Python 2
# para processamento de linguagem natural (NLP)
#
# primeiro criamos um vocabulario de palavras e seus respectivos sentimentos
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
##
def word_feats(words):
    return dict([(word, True) for word in words])
##
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
train_set = negative_features + positive_features + neutral_features
train_set   #mostra o conjunto de treino
##
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
##
# Definimos o classificador. O algoritmo NaiveBayes é bem útil para isso.
classifier = NaiveBayesClassifier.train(train_set) 
# Previsao
neg = 0
pos = 0
sentence = "Awesome movie, I liked it"
sentence = sentence.lower()
words = sentence.split(' ')
# 
# Vejamos como são as sentenças e palavras
sentence
words
##
for word in words:
    classResult = classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1

print 'Positive: ' + str(float(pos)/len(words))
print 'Negative: ' + str(float(neg)/len(words))
##
# Usando Twitter e NLTK
# Vamos definir os twitters positivos e negativos
pos_tweets = [('I love this car', 'positive'),('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]
neg_tweets = [('I do not like this car', 'negative'),('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]
# Vamos colocar os twitters em um tuple
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]  #apply lower() and split if word>3
    tweets.append((words_filtered, sentiment))

tweets
#####
# Para testar nosso modelo, vamos criar um texto e aplicar 
test=[('I feel happy this morning','positive'),('Larry is my friend','positive'),('I do not like that man','negative'),
      ('My house is not great','negative'),('Your song is annoying','negative')]
test_tweets=[]
for (words, sentiment) in test:
    w_filtered = [e.lower() for e in words.split() if len(e) >= 3]  #apply lower() and split if word>3
    test_tweets.append((w_filtered, sentiment))

test_tweets
#
# extraindo as características
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist) 
    word_features = wordlist.keys()
    return word_features
##
word_features = get_word_features(get_words_in_tweets(tweets))
word_features
#
# A função abaixo extrai as caracteristicas que estao no documento e compara com as palavras
def extract_features(document):
    document_words = set(document)
    features = {}   #create a dictionary
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
##
doc=['love','this','car']
fd=extract_features(doc)
###
###
## Criando um classifier
from nltk.classify.util import apply_features
training_set = nltk.classify.apply_features(extract_features, tweets)
training_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(32)
### 
## Vamos usar em um twitter novo
new='Lary is my friend'
new_s=new.split()
new_s
new_e=extract_features(new_s)
new_e
print classifier.classify(new_e)
###############