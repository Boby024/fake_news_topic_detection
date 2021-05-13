import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

modals = ['can', 'could', 'may', 'might', 'must', 'will', 'would', 'should']
stop_words = set(stopwords.words('english') + modals)

def cleaned_text(text):
    pattern = r'\b[^\d\W]+\b'
    tokenizer = RegexpTokenizer(pattern)

    # clean and tokenize document string
    text = str(text).lower()
    text = re.sub('[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    tokens = tokenizer.tokenize(text)

    # remove stop words from tokens
    tokens_out_stop_words = [raw for raw in tokens if not raw in stop_words]
    
    # lemmatize tokens
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(tokens, "v") for tokens in tokens_out_stop_words]
    
    # remove word containing only single char
    tokens = [raw for raw in tokens if not len(raw) == 1]

    # remove duplicate
    tokens = list(set(tokens))

    return " ".join(tokens)

def cleaned_dataset(filename, columns, set_type=True):
    df = pd.read_csv(filename, error_bad_lines=False)
    if set_type == False:
        df = df[df[ [columns[0], columns[1], columns[2] ] ].notnull()]
    else:
        df = df[df[ [columns[0], columns[1], columns[2], columns[3] ] ].notnull()]

    df["title_content"] = df[columns[1]] + " " + df[columns[2]]
    df["cleaned_text"] = df["title_content"].astype(str).apply(lambda x: cleaned_text(x))
    return df

def prediction(train_set, test_set, label_column_name):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(train_set['cleaned_text'], train_set[str(label_column_name)])
    labels = model.predict(test_set['cleaned_text'])
    return labels

def generate_output(filename, test_set, test_set_predicted, old_labe_column_name, new_label_column_name):
    test_set[str(old_labe_column_name)] = test_set_predicted
    test_set.rename(columns= {str(old_labe_column_name): str(new_label_column_name)}, inplace=True)
    header = ["public_id", str(new_label_column_name)]
    test_set.to_csv(filename, columns = header, index=False)




## you can change these paths according where you dataset is (on the file "properties.json")

path_fake_detection = None
path_article_categorization = None
try:
    with open("properties.json") as json_file:
        data = json.load(json_file)
        path_fake_detection = data["fake_news_detection"]
        path_article_categorization = data["article_categorization"]
except Exception as err:
    print(err)

#####################################
#   classifier fake new detection   #
#####################################

if path_fake_detection:
    ## you can change these paths according, where you dataset is
    train_filename =  path_fake_detection["train_path"]
    test_filename = path_fake_detection["test_path"]

    ## task3a
    columns_train_set = ["public_id", "title", "text", "our rating"]
    # get train set

    train_set = cleaned_dataset(train_filename, columns_train_set)
    # get test set

    columns_test_set = ["public_id", "title", "text"]
    test_set = cleaned_dataset(test_filename, columns_test_set, False)

    #result = prediction(train_set, test_set, columns_train_set[-1])
    generate_output("classifier_fake_news_detection_output.csv", test_set, prediction(train_set, test_set, columns_train_set[-1]), "our rating" , "predicted_rating")
    print("Classifier -> Multi-class fake news detection of news articles (English) -> csv file generated")


#########################################
#   classifier article categorization   #
#########################################

if path_article_categorization:
    train_filename =  path_article_categorization["train_path"]
    test_filename = path_article_categorization["test_path"]

    ## task3a
    columns_train_set = ["public_id", "title", "text", "domain"]
    # get train set

    train_set = cleaned_dataset(train_filename, columns_train_set)
    # get test set

    columns_test_set = ["public_id", "title", "text"]
    test_set = cleaned_dataset(test_filename, columns_test_set, False)

    #result = prediction(train_set, test_set, columns_train_set[-1])
    generate_output("classifier_topical_domain_classification_output.csv", test_set, prediction(train_set, test_set, columns_train_set[-1]), "domain" , "predicted_domain")
    print("Classifier -> Topical Domain Classification of News Articles (English)  -> csv file generated")