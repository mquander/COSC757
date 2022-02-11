from  nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re
from nltk.tokenize import word_tokenize
import nltk
#pip install WordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import operator

from sklearn.metrics import accuracy_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import spacy
from itertools import islice
from datetime import date

articles_df = pd.read_json('combined.json', lines=True)

stemmer = SnowballStemmer('english')
stop_words = stopwords.words("english")
stop_words.append('the')

articles_df['cleaned content'] = articles_df['contents'].apply(lambda x: " ".join([stemmer.stem(i) 
                                    for i in re.sub("[^a-zA-Z]", " ", x).split() 
                                        if i not in stop_words]).lower())


year2009_df = articles_df[articles_df['date'].str.contains("2009")]
year2010_df = articles_df[articles_df['date'].str.contains("2010")]
year2011_df = articles_df[articles_df['date'].str.contains("2011")]
year2012_df = articles_df[articles_df['date'].str.contains("2012")]
year2013_df = articles_df[articles_df['date'].str.contains("2013")]
year2014_df = articles_df[articles_df['date'].str.contains("2014")]
year2015_df = articles_df[articles_df['date'].str.contains("2015")]
year2016_df = articles_df[articles_df['date'].str.contains("2016")]
year2017_df = articles_df[articles_df['date'].str.contains("2017")]
year2018_df = articles_df[articles_df['date'].str.contains("2018")]

year2017_df.drop('contents', inplace=True, axis=1)

year2009_df = year2009_df.reset_index()
year2010_df = year2010_df.reset_index()

year2011_df = year2011_df.reset_index()
year2012_df = year2012_df.reset_index()
year2013_df = year2013_df.reset_index()
year2014_df = year2014_df.reset_index()

year2015_df = year2015_df.reset_index()
year2016_df = year2016_df.reset_index()
year2017_df = year2017_df.reset_index()
year2018_df = year2018_df.reset_index()
#year2017_df.loc['cleaned components']

doj_entities = ['Antitru', 'Civil', 'Crimin', 'Environ', 'Security', 'Tax']

for i in range(len(articles_df)):
    if(len(articles_df.at[i, 'components']) > 0):
        articles_df.loc[i, 'cleaned components'] = articles_df.at[i, 'components'][0]

#************** 2009 freqDist of components****************************
freqDist={}
#print(year2009_df.head())
componentsSeries = year2009_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2009 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2009=nltk.FreqDist(freqDist)
fd_2009.plot(10, title='2009 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2010 freqDist of components****************************
freqDist={}
#print(year2010_df.head())
componentsSeries = year2010_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2010 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2010=nltk.FreqDist(freqDist)
fd_2010.plot(10, title='2010 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2011 freqDist of components****************************
freqDist={}
#print(year2011_df.head())
componentsSeries = year2011_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2011 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2011=nltk.FreqDist(freqDist)
fd_2011.plot(10, title='2011 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2012 freqDist of components****************************
freqDist={}
#print(year2012_df.head())
componentsSeries = year2012_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(len(component) > 0 and word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2012 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2012=nltk.FreqDist(freqDist)
fd_2012.plot(10, title='2012 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2013 freqDist of components****************************
freqDist={}
#print(year2013_df.head())
componentsSeries = year2013_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(len(component) > 0 and word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2013 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2013=nltk.FreqDist(freqDist)
fd_2013.plot(10, title='2013 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2014 freqDist of components****************************
freqDist={}
#print(year2014_df.head())
componentsSeries = year2014_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(len(component) > 0 and word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2014 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2014=nltk.FreqDist(freqDist)
fd_2014.plot(10, title='2014 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2015 freqDist of components****************************
freqDist={}
#print(year2015_df.head())
componentsSeries = year2015_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2015 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2015=nltk.FreqDist(freqDist)
fd_2015.plot(10, title='2015 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2016 freqDist of components****************************
freqDist={}
#print(year2016_df.head())
componentsSeries = year2016_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

i=0
for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2016 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2016=nltk.FreqDist(freqDist)
fd_2016.plot(10, title='2016 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2017 freqDist of components****************************
freqDist={}
#print(year2017_df.head())
componentsSeries = year2017_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

#exit(0)
i=0
for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2017 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2017=nltk.FreqDist(freqDist)
fd_2017.plot(10, title='2017 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#************** 2018 freqDist of components****************************
freqDist={}
#print(year2018_df.head())
componentsSeries = year2018_df['components'] # series of components
print("componentsSeries length ", len(componentsSeries))

i=0
for i in (range(len(componentsSeries))):
    component = componentsSeries[i]

    for word in doj_entities:
        if(word in component[0]):
             # stringWord in doj_entities
            if freqDist.__contains__(word):
                 #if : component in freqDist
                freqDist[word] += 1
            else:
                freqDist[word] = 1


sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print("2018 freqDist: ", freqDist)

plt.figure(figsize=(12, 8))
fd_2018=nltk.FreqDist(freqDist)
fd_2018.plot(10, title='2018 Frequency Distribution of Divisions', cumulative=False) 

#*****************************************************************

#*************working frequency distribution of doj entities in contents************
print("year2017 dataframe head: \n", year2017_df.head())
print("year2017 length: \n", len(year2017_df))
words_2017 = year2017_df['cleaned content'].str.cat(sep='') #joins string of every row
print("words_2017 type: ", type(words_2017))
tok_2017 = word_tokenize(words_2017)
i=0
nullComponents = 0
for  i in range(len(year2017_df)):
    contentWords = year2017_df.at[i, 'cleaned content']
    if year2017_df.at[i, 'components'] == '':
        nullComponents+=1
    contentTokens = word_tokenize(contentWords)
    for stringWord in contentTokens: # and word in doj_entities:
        for word in doj_entities:
            if(word in stringWord): # stringWord in doj_entities
                if freqDist.__contains__(word):
                    freqDist[word] += 1
                else:
                   freqDist[word] = 1

sortedFreq = sorted(freqDist.items(), key=operator.itemgetter(1), reverse=True)
print(freqDist)
print("null components: ", nullComponents)
plt.figure(figsize=(12, 8))
fd_2017=nltk.FreqDist(freqDist)
fd_2017.plot(10, cumulative=False) 

# *************************************

print("year2009 length: \n", len(year2015_df))
words_2015 = year2015_df['cleaned content'].str.cat(sep='') #joins string of every row
print("words_2009 type: ", type(words_2015))

tok_2015 = word_tokenize(words_2015)

tok_2015 = [word for word in tok_2015 if (len(word) > 3 and not( word.__contains__("attor") 
                                                           or word.__contains__("assist")  
                                                           or word.__contains__("depart") 
                                                           or word.__contains__("us")  
                                                           or word.__contains__("unit")
                                                           or word.__contains__("state")
                                                           or word.__contains__("general")
                                                           or word.__contains__("includ")
                                                           or word.__contains__("justic")
                                                           or word.__contains__("plead")
                                                           or word.__contains__("guilt")
                                                           or word.__contains__("special")
                                                           or word.__contains__("agent")
                                                           or word.__contains__("crimin")
                                                           or word.__contains__("divis")
                                                           or word.__contains__("victim")
                                                           or word.__contains__("prison")
                                                           or word.__contains__("eastern")
                                                           or word.__contains__("deputi")
                                                           or word.__contains__("supervis")
                                                           or word.__contains__("distric")
                                                           or word.__contains__("compani")
                                                           ))]

print("tok_2009 type: ", type(tok_2015))
print("tok_2009 length: ", len(tok_2015))

plt.figure(figsize=(10,5))
wc = WordCloud(max_font_size=50, max_words=100, background_color="white")
wordcloud_2015 = wc.generate_from_text(' '.join(tok_2015))

plt.imshow(wordcloud_2015, interpolation="bilinear")
plt.axis("off")
plt.show()

#    TF-IDF processing
#**********************************************

tfIdfTransformer = TfidfTransformer(use_idf=True)
countVectorizer = CountVectorizer()
wordCount = countVectorizer.fit_transform(year2010_df['cleaned content'])
newTfIdf = tfIdfTransformer.fit_transform(wordCount)
df = pd.DataFrame(newTfIdf[0].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF 2010"])
df = df.sort_values('TF-IDF 2010', ascending=False)
print (df.head(25))


# TF-IDF values for words in 2017 contents
#year2018_df = articles_df[articles_df['date'].str.contains("2018")]
doj_ents = ['Antitrust Division', 'Civil Division', 'Civil Rights Division', 'Criminal Division', 'Environmental and Natural Resources Division', 'National Security Division', 'Tax Division']
sub_df = articles_df.loc[articles_df['cleaned components'].isin(doj_ents)]


sub_df = sub_df.reset_index()
sub_df.drop('components', inplace=True, axis=1)

print("sub_df size: ", len(sub_df))
print("sub_df head:\n ", sub_df.head())



train_X, test_X, train_Y, test_Y = model_selection.train_test_split(sub_df['cleaned content'], sub_df['cleaned components'], test_size=0.3)
encoder = LabelEncoder()
#print(train_X[0:12])

train_Y = encoder.fit_transform(train_Y)
test_Y = encoder.fit_transform(test_Y)
print("train_Y type: ", type(train_Y))
print("train_Y head: ", train_Y[0:22])

tfIdfVectorizer=TfidfVectorizer() #use_idf=True
tfIdfVectorizer.fit(sub_df['cleaned content'])

train_X_tfidf = tfIdfVectorizer.transform(train_X)
test_X_tfidf = tfIdfVectorizer.transform(test_X)


print("train_X_tfidf type: ", type(train_X_tfidf))
#print("train_X_tfidf 12:: ", train_X_tfidf[12:])
#print(train_X_tfidf.sort())  1st num is the row number in train_X_tfidf, 2nd num is unique int of word, 3rd num is tf-idf score

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_X_tfidf, train_Y) 
predictions_SVM = SVM.predict(test_X_tfidf) # predict the labels on validation dataset
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, test_Y)*100) # Use accuracy_score function to get the accuracy

cleanedContent = year2018_df['cleaned content']
print("cleanContent length: ", len(cleanedContent))
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(cleanedContent)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))


print("End")