import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def classify(train_file, test_file):
    # todo: implement this function

    classes = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5'] #This is y

    train_review_dicts = extract_review_dictionaries(train_file)
    test_review_dicts = extract_review_dictionaries(test_file)

    train_data, train_target = extract_data_and_target(train_review_dicts)
    test_data, test_target = extract_data_and_target(test_review_dicts)

    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    model.fit(train_data, train_target)

    labels = model.predict(test_data)
    c_mat = confusion_matrix(test_target, labels)
    sns.heatmap(c_mat.T, square = True, annot = True, fmt = 'd', cbar = False, xticklabels = classes, yticklabels = classes)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # todo: you can try working with various classifiers from sklearn:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    #  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #  please use the LogisticRegression classifier in the version you submit

    # todo: fill in the dictionary below with actual scores obtained on the test data
    test_results = {'class_1_F1': 0.0,
                    'class_2_F1': 0.0,
                    'class_3_F1': 0.0,
                    'class_4_F1': 0.0,
                    'class_5_F1': 0.0,
                    'accuracy': 0.0}

    return test_results


def extract_review_dictionaries(review_file):
    review_dict_list = []

    with open(review_file) as rf:
        for review_json_obj in rf:
            review_dict = json.loads(review_json_obj)
            review_dict_list.append(review_dict)

    return review_dict_list


def extract_data_and_target(review_dict_list):
    data = []
    target = []

    for review_dict in review_dict_list:
        review_text = review_dict.get("reviewText", "")
        review_overall = review_dict.get("overall")
        data.append(review_text)
        target.append(review_overall - 1)

    return (data, target)

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
