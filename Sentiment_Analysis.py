import nltk
#nltk.download('all')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importing the dataset
columns=['TweetID', 'Topic', 'Target', 'Text']
train = pd.read_csv('twitter_training.csv', names = columns)
test = pd.read_csv('twitter_validation.csv', names = columns)
dataset = pd.concat([train, test], ignore_index = False)
dataset.head()
dataset.describe(include = 'object')
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)

# Visualisation
dataset['Topic'].value_counts().plot(kind = 'bar')
plt.figure(figsize=(9, 7))
crosstab = pd.crosstab(index=dataset['Topic'], columns=dataset['Target'])
sns.heatmap(crosstab, cmap = 'crest')
corpus = ' '.join(dataset['Text'])
wc = WordCloud(width=1200, height=500).generate(corpus)
plt.imshow(wc, interpolation='bilinear')

# Data Cleaning
text = dataset['Text']
dataset['Text'] = dataset['Text'].astype(str)
l = []
text = dataset['Text']
for t in text:
    if type(t) not in l:
        l.append(type(t))
        
# Tokenization
modified_text = []
rows = len(text)
for iText in dataset['Text']:
    iText = iText.lower()
    iText = re.sub(r'[^\w\s]', '', iText)
    iText = re.sub(r'\d+', '', iText)
    tokens = word_tokenize(iText)
    words = set(stopwords.words('english'))
    doc = [word for word in tokens if word not in words]
    finalText = ' '.join(doc)
    modified_text.append(finalText) 

dataset.drop('Text', axis =1 , inplace = True)
dataset['Text'] = modified_text
dataset.head()

# Machine Learning Model
X = dataset['Text'] 
Y = dataset['Target'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Vectorizer = TfidfVectorizer()
X_train_tfidf = Vectorizer.fit_transform(X_train)
X_test_tfidf = Vectorizer.transform(X_test)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_tfidf, Y_train)
Y_pred = rf_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, Y_pred)
print("Test Accuracy:", accuracy)
print(classification_report(Y_test, Y_pred))
