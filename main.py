import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""## Load Dataset From JSON"""

# the function loads dataset from JSON and returns data to be trained on and the label - review rating in our case 
def load_dataset(ds_file):
  reviews = [] 
  y = [] 
  with open(ds_file) as f: 
    for review in f: 
        reviewDict = json.loads(review)
        reviews.append(reviewDict.get("reviewText","") + reviewDict.get("summary",""))       
        y.append(int(reviewDict["overall"])-1)
  return np.array(reviews), np.array(y)

"""## Get F1 score and Test Metrics """

def display_confusionMatrix(confusion_mat):
  labels = ["1 star", "2 stars", "3 stars", "4 starts", "5 stars"]
  sns.heatmap(confusion_mat.T, square = True, annot = True, fmt = 'd', cbar = False, xticklabels = labels, yticklabels = labels, cmap='Blues')
  plt.xlabel('true label')
  plt.ylabel('predicted label')


def get_metrics(target,predicted, displayConfusionMatrix = False):
  # filling in the dictionary below with actual scores obtained on the test data
  test_results = {'class_1_F1': 0.0,
                  'class_2_F1': 0.0,
                  'class_3_F1': 0.0,
                  'class_4_F1': 0.0,
                  'class_5_F1': 0.0,
                  'accuracy': 0.0}
  test_results["accuracy"] = np.mean(predicted == target)       
  f1_scores = f1_score(target,predicted, average=None)
  for i in range(1,f1_scores.shape[0]+1):
    key = f'class_{i}_F1' 
    test_results[key] = f1_scores[i-1]
  if displayConfusionMatrix: 
    cm = confusion_matrix(target, predicted)
    display_confusionMatrix(cm)

  return test_results

"""###Optional List of Stop Words"""

stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']

def classify(train_file, test_file):
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')
    

    train_data , target = load_dataset(train_file)
   
        
    K = 1000 # Size of count vector (leave only the most frequent terms)
    # optional parameters for CountVectorizer include stop_words - remove stop words from corpus, max_features - limit Vocabulary to K most freq n-grams/words
    model = Pipeline([('vect', CountVectorizer(ngram_range=(1,3),max_features=K)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=8, tol=None)),])
    model.fit(train_data, target)


    # loading test dataset..
    test_data ,test_target = load_dataset(test_file)  
    predicted = model.predict(test_data)    
    test_results = get_metrics(test_target, predicted, True)                  

    return test_results


"""##5. Extraction of K-Best Features"""

def print_k_best_features(dataset_file):
  train_data , target = load_dataset(dataset_file)  
  count_vect = CountVectorizer(ngram_range=(1,3), stop_words=stop_words)
  counts=count_vect.fit_transform(train_data)
  select = SelectKBest(score_func=chi2, k=15)
  select.fit_transform(counts,target)
  k_best=np.asarray(count_vect.get_feature_names_out())[select.get_support()]
  print(k_best)


if __name__ == '__main__':
  with open('config.json', 'r') as json_file:
    config = json.load(json_file)

  plt.figure()
  plt.title(f'Confusion Matrix on {config["test_data"]}')
  results = classify(config['train_data'], config['test_data'])

  for k, v in results.items():
    print(k, v)

  print_k_best_features(config['train_data'])

  """## 6.Cross Domain Classification"""

  # now we are going to predict on the Automotive reviews using a model trained on pets reviews 
  test_file = "data/Automotive.test.json"
  plt.figure()
  plt.title(f'Cross-Domain Classification on {test_file}')
  test_results = classify(config['train_data'], test_file)

  for k, v in results.items():
    print(k, v)

  # """## Creating a model trained on the other datasets"""

  # config = {"train_data" : "data/Automotive.train.json",
  #           "test_data" : "data/Automotive.test.json"}
  # results = classify(config['train_data'], config['test_data'])
  # for k, v in results.items():
  #   print(k, v)

  # config = {"train_data" : "data/Sports_and_Outdoors.train.json",
  #           "test_data" : "data/Sports_and_Outdoors.test.json"}
  # results = classify(config['train_data'], config['test_data'])

  # for k, v in results.items():
  #   print(k, v)

  plt.show()