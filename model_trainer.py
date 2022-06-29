from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import pickle

product_corpus_df = pd.read_csv('shopmania_cat.csv')

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(product_corpus_df['category'])
product_corpus_df['label'] = label_encoder.transform(product_corpus_df['category'])

# I refrained from using the stopword parameter for the vectorizer. I read a research paper which said better results
# can be achieved if you use full corpus when the input text is limited i.e. titles are not very long so use all words
# for classification. If your titles are very big use vectorizer = TfidfVectorizer(stop_words='English')
vectorizer = TfidfVectorizer()

x = product_corpus_df['title']
y = product_corpus_df['label']
vectorized_x = vectorizer.fit_transform(x)

# run in parallel on the cpu. fill in your number of threads.
rf_clf = RandomForestClassifier(n_jobs=10)
rf_clf.fit(vectorized_x, y)

pickle.dump(rf_clf, open('product_text_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('product_text_vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open('product_text_encoder.pkl', 'wb'))
