#!/usr/bin/env python
# coding: utf-8

# # History of Phylosophy: More Focused and Less Neutral as Evolving
# 
# ## 1.Data and Packages Importing

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from random import sample
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
df = pd.read_csv('../data/philosophy_data.csv')


# In[2]:


df.head()


# In[3]:


df.groupby(['original_publication_date','school','author']).count()


# From the dataset shown above, it is observed that each row contains school, author, and lemmatized text. Hence we are interested in how does the key word and the sentiment of these text for each school or each author evolves as time goes; furthermore, we may explore if there is also a trend in key word and sentiment of text from various author.
#     
# ## 2 Analysis of the Schools
# ### 2.1 Wordcould Analysis
# 
# In the cell below, we create a wordcloud image for each school.

# In[4]:


school = df['school'].unique().tolist()
stop_words = set(stopwords.words('english'))
plt.figure(figsize=(20,16)) #set figure size
for s in range(len(school)):
    df_temp = df[df.school==school[s]]
    text = " ".join(txt for txt in df_temp.sentence_lowered)
    plt.subplot(7,2,s+1).set_title(school[s])
    wordcloud = WordCloud(stopwords=stop_words,background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
plt.show()


# It is obvious that the phylosophy in the early stage tend to focus more on human self, and they have a lot of high frequency words where people hardly grasp the key idea from the world cloud. They tend to use words such as one, thing, and say. Some other words with high frequencies are relatively less related to the core topic of the specific school. However, as time involves, philosophy are more related to other realms when sociology and economics developed rapidly. Each school is more focused on one specific problem instead of being general. Examples of this trend are capitalism, communism, and feminism. One of the characteristics observed from the plots above is that the number of high frequency word is fewer than early schools. Also, it is easier to identify these schools from others as they have a clear focus such as labour, capital value, or women rights.
#     
# ### 2.2 Sentiment Analysis
# 
# The cell below generate a barchart of sentiment for each school.

# In[5]:


school_score = []
school1 = []
senti = []
senti1= ['neg','neu','pos'] 
for s in range(len(school)):
    df_temp = df[df.school==school[s]]
    text_list = df_temp.lemmatized_str.to_list()
    list1 = sample(text_list,min(1000,len(text_list)))
    text = ' '.join(list1) #text of the school
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    for i in range(len(senti1)):
        school1.append(school[s])
        school_score.append(score.get(senti1[i]))
    senti.extend(senti1)
school_df = pd.DataFrame(zip(school1,school_score,senti),columns=['school','score','sentiment'])
plt.figure(figsize=(8,18))
sns.barplot(x='score',y='school',hue='sentiment', data=school_df) # generate barplot for each school
plt.show()


# Theoretically speaking, philosophy tend to have neutral sentiment in most cases. However, there are still observations if we consider the trend. Most of the schools have high neutral score, also, they tend to have more negative sentiment then positive as they are crticizing most of the time. There are also exceptions. Rationalizion, nietzsche, and feminism have expecially low neutrak score and high negative score. This is caused by their inherent pessimism style such as rationalism find that people fail to be rational at the end and feminism are mostly criticizing the current social situation. Generally speaking, the later schools are more likely to have a less neutral statement style than the early schools.
# 
# ## 3. Analysis on Authors
# ### 3.1 Word Could Analysis
# 
# After examine the schools, we may see the authors and see if similar trends continues to exist. The following cell demonstrate the wordcloud for each author.

# In[6]:


author = df['author'].unique().tolist()
stop_words = set(stopwords.words('english'))
plt.figure(figsize=(20,16))
for a in range(len(author)):
    df_temp = df[df.author==author[a]]
    text = " ".join(txt for txt in df_temp.sentence_lowered)
    plt.subplot(12,3,a+1).set_title(author[a])
    wordcloud = WordCloud(stopwords=stop_words,background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
plt.show()


# From the word cloud above, the trend follows as discribed in 2.1. Early philosopher tend to have more general topic with less focus. They are more likely to discuss human-being and mindset instead of exterior topics. As time involves, philosopher are getting more focused on a smmall topic such as human-being, human rights, or economics. In this stage, people such as Marx and Wollstonecraft are getting a clear focus on specific word as shown on the word cloud which is also the key of corresponding school.
# 
# ### 3.2 Sentiment Analysis
# 
# The cell below generate a barplot demontrating sentiment analysis on each author's text.

# In[7]:


author_score = []
author1 = []
senti = []
senti1 = ['neg','neu','pos']
for a in range(len(author)):
    df_temp = df[df.author==author[a]]
    text_list = df_temp.lemmatized_str.to_list()
    list1 = sample(text_list,min(1000,len(text_list)))
    text = ' '.join(list1)
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    for i in range(len(senti1)):
        author1.append(author[a])
        author_score.append(score.get(senti1[i]))
    senti.extend(senti1)
author_df = pd.DataFrame(zip(author1,author_score,senti),columns=['author','score','sentiment'])
plt.figure(figsize=(8,28))
sns.barplot(x='score',y='author',hue='sentiment', data=author_df) #generate bar plot for each author
plt.show()


# The plots above further proves the conclusion in 2.2. Philosophers are generally getting less neutral and more negative in as time gies. One exception in the plot are Malebranche, Spinoza, and Lebniz who are author of rationalism. In later eras, phiosopher such as Nietzsche and Wollstonecraft are becoming much less neutral than early phylosophers.
# 
# ## 3. Conclusion
#     
# From all the analysis above, we can draw the conclusion that phylosophy is getting more focused and less neutral as it evolves. It is understandable as people are belonging to a more specific school and have clearer view to specific problems.
