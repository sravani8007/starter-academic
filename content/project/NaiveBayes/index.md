---
title: Naive Bayes Classifier
summary: In this project i calculated naive bayes classifier for a large text dataset(largest movie review).
        I calculated conditional probability for all the words and used it to determine the sentiment of review
        In this process i verified the results by applying five-fold cross validation along with laplace smoothing.
tags:
- Python
date: "2016-04-27T00:00:00Z"
external_link: ""


#links:
#- icon: twitter
url_code: "https://www.dropbox.com/s/rebimsmum5dc2hy/Suravajhula_03.ipynb?dl=1"
#url_pdf: "files/Suravajhula_03.ipynb"
#url_slides: "./Suravajhula_03.ipynb"
#url: "files/Suravajhula_03.ipynb"
#url_video: ""

---

### Assignment 3

Name: Sravani Suravajhula
UTA Id:1001778007

In this project i implemented Naive Bayes Classifier from scratch in python.
for this assignment i used text dataset about movie review(http://ai.stanford.edu/~amaas/data/sentiment/).


```python
import itertools
import re
import copy
import random 
import numpy as np
import pandas as pd
from collections import defaultdict,Counter
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords



```


```python
#random.seed(0)
print_data=1

train_neg=load_files('aclImdb/train', categories= ['neg'],encoding='utf-8')
train_pos=load_files('aclImdb/train', categories= ['pos'],encoding='utf-8')
test_neg=load_files('aclImdb/test', categories= ['neg'],encoding='utf-8')
test_pos=load_files('aclImdb/test', categories= ['pos'],encoding='utf-8')


```

## preprocessing the data


```python
def cleanup_data(data):
    pattern1=re.compile("[!#$%&'()*+,\'\"-./:;<=>?@[\]^_`{|}~]")
    pattern2=re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    data=[re.sub(pattern1, '', line) for line in data]
    data=[re.sub(pattern2, ' ', line).lower() for line in data]
    return data
    
train_neg_data=cleanup_data(train_neg['data'])
train_pos_data=cleanup_data(train_pos['data'])
test_neg_data=cleanup_data(test_neg['data'])
test_pos_data=cleanup_data(test_pos['data'])
```

## Building vocabulary as list


```python
train_X=train_neg_data+train_pos_data
train_Y=[0 for i in range(len(train_neg_data))]+[1 for i in range(len(train_pos_data))]

test_X=test_neg_data+test_pos_data
test_Y=[0 for i in range(len(test_neg_data))]+[1 for i in range(len(test_pos_data))]
#print(train_X[0])
train_X=[word.split() for word in train_X ]
test_X=[word.split() for word in test_X ]

print((train_X[0]))
print(len(train_X),len(train_Y))


```

    ['so', 'theres', 'an', 'old', 'security', 'guard', 'and', 'a', 'guy', 'who', 'dies', 'and', 'then', 'theres', 'kevin', 'the', 'worlds', 'biggest', 'wuss', 'kevin', 'wants', 'to', 'impress', 'his', 'incredibly', 'insensitive', 'bratty', 'and', 'virginal', 'girlfriend', 'amy', 'as', 'he', 'returns', 'from', 'work', 'to', 'a', 'random', 'house', 'he', 'finds', 'his', 'friends', 'the', 'sexually', 'confusing', 'redshorted', 'kyle', 'and', 'the', 'truly', 'revolting', 'sluttish', 'daphne', 'they', 'are', 'soon', 'joined', 'by', 'daphnes', 'boyfriend', 'the', 'triggerhappy', 'sexcrazed', 'macho', 'lunkhead', 'nick', 'and', 'theres', 'the', 'title', 'creatures', 'horrid', 'little', 'dogeared', 'puppets', 'who', 'kill', 'people', 'by', 'giving', 'them', 'their', 'hearts', 'desire', 'kyles', 'hearts', 'desire', 'is', 'to', 'mate', 'with', 'a', 'creepy', 'yucky', 'woman', 'in', 'spandex', 'nicks', 'hearts', 'desire', 'is', 'to', 'throw', 'grenades', 'in', 'a', 'grade', 'school', 'cafeteria', 'i', 'mean', 'nightclub', 'kevins', 'hearts', 'desire', 'is', 'to', 'beat', 'up', 'a', 'skinny', 'thug', 'with', 'nunchucks', 'amys', 'hearts', 'desire', 'is', 'to', 'be', 'a', 'disgusting', 'slut', 'daphnes', 'already', 'a', 'disgusting', 'slut', 'so', 'she', 'doesnt', 'have', 'a', 'hearts', 'desire', 'along', 'the', 'way', 'a', 'truly', 'hideous', 'band', 'sings', 'a', 'truly', 'odd', 'song', 'the', 'hobgoblins', 'randomly', 'go', 'back', 'to', 'where', 'they', 'came', 'from', 'then', 'blow', 'up', 'citizen', 'kane', 'cannot', 'hold', 'a', 'candle', 'to', 'this', 'true', 'masterpiece', 'of', 'american', 'cinema']
    25000 25000
    

## Dividing the dataset into train,development and test 


```python
temp=random.randint(0,100)
train_data_x,dev_data_x,train_data_y,dev_data_y=train_test_split(train_X,train_Y,test_size=0.2,random_state=temp)
flatten = itertools.chain.from_iterable
complete_data=list(flatten(train_data_x))
complete_counter=Counter(complete_data)
#print(complete_counter['the'])
sorted_words=dict(sorted(complete_counter.items(),key=lambda i:i[1],reverse=True))
print('number of total words before omitting the uncommon words:',len(sorted_words))
for word in list(sorted_words):
    if sorted_words[word] <5:
        del sorted_words[word]
print('number of total words after ommiting the uncommon words:',len(sorted_words))       
required_words=list(sorted_words.keys())



```

    number of total words before omitting the uncommon words: 107625
    number of total words after ommiting the uncommon words: 28582
    

## counting number of words in a document


```python
#print(train_data_x[0])

train_data_x_cnted=[dict(Counter(word)) for word in train_data_x ]
print(train_data_x_cnted[0])
```

    ['the', 'film', 'was', 'okay', 'quite', 'entertaining', 'the', 'cast', 'was', 'pretty', 'good', 'and', 'ill', 'second', 'what', 'the', 'comment', 'before', 'me', 'mentioned', 'glenn', 'quinn', 'was', 'outstanding', 'and', 'he', 'alone', 'is', 'reason', 'enough', 'to', 'watch', 'this', 'movie', 'he', 'played', 'the', 'selfish', 'evil', 'friend', 'and', 'manager', 'of', 'the', 'band', 'brilliantlybr', 'br', 'there', 'are', 'a', 'lot', 'of', 'songs', 'performed', 'by', 'beyond', 'gravity', 'in', 'this', 'film', 'but', 'this', 'doesnt', 'really', 'come', 'as', 'a', 'surprise', 'considering', 'the', 'film', 'is', 'a', 'vh1', 'production', 'however', 'if', 'the', 'soft', 'rock', 'pop', 'music', 'isnt', 'to', 'someones', 'liking', 'one', 'might', 'as', 'well', 'flash', 'forward', 'those', 'scenesbr', 'br', 'the', 'plot', 'of', 'a', 'band', 'trying', 'to', 'make', 'it', 'to', 'the', 'top', 'in', 'la', 'but', 'having', 'to', 'overcome', 'many', 'obstacles', 'on', 'the', 'way', 'isnt', 'too', 'original', 'but', 'quite', 'entertaining', 'with', 'some', 'surprising', 'plot', 'turns', 'here', 'and', 'there']
    {'the': 10, 'film': 3, 'was': 3, 'okay': 1, 'quite': 2, 'entertaining': 2, 'cast': 1, 'pretty': 1, 'good': 1, 'and': 4, 'ill': 1, 'second': 1, 'what': 1, 'comment': 1, 'before': 1, 'me': 1, 'mentioned': 1, 'glenn': 1, 'quinn': 1, 'outstanding': 1, 'he': 2, 'alone': 1, 'is': 2, 'reason': 1, 'enough': 1, 'to': 5, 'watch': 1, 'this': 3, 'movie': 1, 'played': 1, 'selfish': 1, 'evil': 1, 'friend': 1, 'manager': 1, 'of': 3, 'band': 2, 'brilliantlybr': 1, 'br': 2, 'there': 2, 'are': 1, 'a': 4, 'lot': 1, 'songs': 1, 'performed': 1, 'by': 1, 'beyond': 1, 'gravity': 1, 'in': 2, 'but': 3, 'doesnt': 1, 'really': 1, 'come': 1, 'as': 2, 'surprise': 1, 'considering': 1, 'vh1': 1, 'production': 1, 'however': 1, 'if': 1, 'soft': 1, 'rock': 1, 'pop': 1, 'music': 1, 'isnt': 2, 'someones': 1, 'liking': 1, 'one': 1, 'might': 1, 'well': 1, 'flash': 1, 'forward': 1, 'those': 1, 'scenesbr': 1, 'plot': 2, 'trying': 1, 'make': 1, 'it': 1, 'top': 1, 'la': 1, 'having': 1, 'overcome': 1, 'many': 1, 'obstacles': 1, 'on': 1, 'way': 1, 'too': 1, 'original': 1, 'with': 1, 'some': 1, 'surprising': 1, 'turns': 1, 'here': 1}
    

## Calculating the probability of occurence of a word and
## conditional probability based on sentiment


```python
#print(train_data_x_cnted)
total_words={}
for i,line in enumerate(train_data_x_cnted):
    #print(line)
    if train_data_y[i]==1:
        sent='pos'
    else:
        sent='neg'
    for wrd in line:
        
        if not total_words.get(wrd) :
            total_words[wrd]={'pos':0,'neg':0}
        total_words[wrd][sent]+=1
        
pos_len=sum(train_data_y) 
print('total positive documents in training:',pos_len)
neg_len=len(train_data_y)-pos_len
print('total negative documents in traing:',neg_len)
print('number of documents containing word "the":',total_words['the'])

final_words={}
for wrd in required_words:
    final_words[wrd]=copy.deepcopy(total_words[wrd])
    final_words[wrd]['total']=final_words[wrd]['pos']+final_words[wrd]['neg']
    final_words[wrd]['pos_prob']=final_words[wrd]['pos']/pos_len
    final_words[wrd]['neg_prob']=final_words[wrd]['neg']/neg_len
    final_words[wrd]['total_prob']=final_words[wrd]['total']/(pos_len+neg_len)
print(final_words['the'])
```

    total positive documents in training: 10042
    total negative documents in traing: 9958
    number of documents containing word "the": {'pos': 9946, 'neg': 9886}
    {'pos': 9946, 'neg': 9886, 'total': 19832, 'pos_prob': 0.9904401513642701, 'neg_prob': 0.9927696324563166, 'total_prob': 0.9916}
    

## Calculating the probability of occurence of a word and
## conditional probability based on sentiment with smoothing


```python

final_words_smooth={}
for wrd in required_words:
    final_words_smooth[wrd]=copy.deepcopy(total_words[wrd])
    final_words_smooth[wrd]['total']=final_words_smooth[wrd]['pos']+final_words_smooth[wrd]['neg']
    final_words_smooth[wrd]['pos_prob']=(final_words_smooth[wrd]['pos']+1)/(pos_len+2)
    final_words_smooth[wrd]['neg_prob']=(final_words_smooth[wrd]['neg']+1)/(neg_len+2)
    final_words_smooth[wrd]['total_prob']=(final_words_smooth[wrd]['total']+1)/(pos_len+neg_len+2)
print('after smoothing:',final_words_smooth['the'])

```

    after smoothing: {'pos': 9946, 'neg': 9886, 'total': 19832, 'pos_prob': 0.9903424930306651, 'neg_prob': 0.9926706827309237, 'total_prob': 0.9915508449155085}
    

## Calculating accuracy using Dev dataset


```python

def nb_main(dev_x,dev_y):
    dev_X_cnted=[list(Counter(word)) for word in dev_x ]
    dev_y_pred=[]
    verify_words=copy.deepcopy(final_words)
    #print(dev_X_cnted[0])
    accurate=0
    for i,words in enumerate(dev_X_cnted):
        pos_prob=1
        neg_prob=1
        for word in words:
            if verify_words.get(word):
                 if verify_words[word]['pos_prob']>0:
                    pos_prob=pos_prob*verify_words[word]['pos_prob']
            if verify_words.get(word):
                if verify_words[word]['neg_prob']>0:
                    neg_prob=neg_prob*verify_words[word]['neg_prob']    
        if pos_prob>neg_prob:
            dev_y_pred.append(1)
        else:
            dev_y_pred.append(0)
        if dev_y[i]==dev_y_pred[i]:
            accurate+=1

    return dev_y_pred,accurate

dev_y_pred,accurate=nb_main(dev_data_x,dev_data_y)
accuracy=(accurate/len(dev_y_pred))*100
print('Development Accuracy',accuracy)
```

    Development Accuracy 75.62
    




```python
def nb_main_smooth(dev_x,dev_y):
    dev_X_cnted=[list(Counter(word)) for word in dev_x ]
    dev_y_pred=[]
    verify_words=copy.deepcopy(final_words_smooth)
    #print(dev_X_cnted[0])
    accurate=0
    for i,words in enumerate(dev_X_cnted):
        pos_prob=1
        neg_prob=1
        for word in words:
            if verify_words.get(word):
                 if verify_words[word]['pos_prob']>0:
                    pos_prob=pos_prob*verify_words[word]['pos_prob']
            if verify_words.get(word):
                if verify_words[word]['pos_prob']>0:
                    neg_prob=neg_prob*verify_words[word]['neg_prob']    
        if pos_prob>neg_prob:
            dev_y_pred.append(1)
        else:
            dev_y_pred.append(0)
        if dev_y[i]==dev_y_pred[i]:
            accurate+=1

    return dev_y_pred,accurate
```

## Calculating accuracy using dev data by conducting five-fold cross validation 


```python

dev_x=[]
dev_y=[]
for i in range(5):
        temp=random.randint(0,100) 
        temp_train_x,temp_dev_x,temp_train_y,temp_dev_y=train_test_split(train_X,train_Y,test_size=0.2,random_state=temp)
       
        dev_x.append(temp_dev_x)
        dev_y.append(temp_dev_y)
 

def cross_validation(dev_list_x,dev_list_y,smooth):
    k=5
    accuracy=[]

    for i in range(5):
        temp_dev_x=dev_list_x[i]
        temp_dev_y=dev_list_y[i]
        if not smooth:
            dev_y_pred,temp_correct=nb_main(temp_dev_x,temp_dev_y)
        else:
            dev_y_pred,temp_correct=nb_main_smooth(temp_dev_x,temp_dev_y)
        temp_accuracy=(temp_correct/len(temp_dev_y))*100
        
        accuracy.append(temp_accuracy)
        
        print('Accuracy for dataset',i+1,' is =',round(temp_accuracy,2),'%')
    return accuracy

devdata_accuracy=cross_validation(dev_x,dev_y,False)
print('five-fold dev data accuracy  without smoothing =',round(sum(devdata_accuracy)/len(devdata_accuracy),2),'%')

```

    Accuracy for dataset 1  is = 75.66 %
    Accuracy for dataset 2  is = 77.06 %
    Accuracy for dataset 3  is = 76.26 %
    Accuracy for dataset 4  is = 76.9 %
    Accuracy for dataset 5  is = 76.62 %
    five-fold dev data accuracy  without smoothing = 76.5 %
    

## calculating five fold Dev data accuracy with smoothing


```python
devdata_accuracy_smooth=cross_validation(dev_x,dev_y,True)
print('five-fold dev data accuracy with smoothing =',round(sum(devdata_accuracy_smooth)/len(devdata_accuracy_smooth),2),'%')
```

    Accuracy for dataset 1  is = 83.18 %
    Accuracy for dataset 2  is = 83.92 %
    Accuracy for dataset 3  is = 83.3 %
    Accuracy for dataset 4  is = 84.26 %
    Accuracy for dataset 5  is = 83.94 %
    five-fold dev data accuracy with smoothing = 83.72 %
    

without smoothing we are ignoring conditinal probability for many words where as with smoothing we are considering all the words in the vocabulary thus we get true conditional probability with more accurate results.
In my experiment this shows clearly as the accuracy of development dataset with five fold cross validation without smoothing is 76.5% With smoothing the accuracy is 83.72%.so, the accuracy improved with smoothing.

## top 10 words that predicts positive and negative classes


```python

#top_pos_pred=dict(sorted(final_words.items(),key=lambda i:i[1]['pos_prob'],reverse=True))
#top_neg_pred=dict(sorted(final_words.items(),key=lambda i:i[1]['neg_prob'],reverse=True))

neg_pred={}
pos_pred={}
stop = set(stopwords.words('english'))
actual_words=[wrd for wrd in final_words.keys() if not wrd in stop]
actual_words=[wrd for wrd in actual_words if wrd.isalpha()]
actual_words=[wrd for wrd in actual_words if len(wrd)>2]
for dic in actual_words:

    
    if final_words[dic]['pos_prob']>final_words[dic]['neg_prob']:
        pos_pred[dic]={'pos_prob':final_words[dic]['pos_prob']}
    if final_words[dic]['neg_prob']>final_words[dic]['pos_prob']:
        neg_pred[dic]={'neg_prob':final_words[dic]['neg_prob']}
    
top_pos_pred=dict(sorted(pos_pred.items() ,key=lambda i:i[1]['pos_prob'],reverse=True))
#print(top_pos_pred)
print('Top 10 Positive words:',list(top_pos_pred.keys())[:10])

top_neg_pred=dict(sorted(neg_pred.items() ,key=lambda i:i[1]['neg_prob'],reverse=True))
#print(top_pos_pred)
print('Top 10 Negative words:',list(top_neg_pred.keys())[:10])
```

    Top 10 Positive words: ['one', 'film', 'time', 'great', 'story', 'see', 'well', 'also', 'first', 'best']
    Top 10 Negative words: ['movie', 'like', 'even', 'good', 'would', 'bad', 'really', 'dont', 'much', 'get']
    

## using test data set calculating final accuracy


```python
test_y_pred,test_correct=nb_main(test_X,test_Y)
test_accuracy=(test_correct/len(test_y_pred))*100

print('Test Accuracy without Smoothing is:',test_accuracy,'%')
```

    Test Accuracy without Smoothing is: 77.388 %
    

## using test data set for calculating final accuracy with smoothing


```python
test_y_pred,test_correct=nb_main_smooth(test_X,test_Y)
test_accuracy=(test_correct/len(test_y_pred))*100

print('Test Accuracy with smoothing is:',test_accuracy,'%')
```

    Test Accuracy with smoothing is: 79.908 %
    

## using five-fold cross validation for final accuracy


```python
test_five_x=[]
test_five_y=[]
for i in range(5):
        temp=random.randint(0,100) 
        _,temp_test_x,_,temp_test_y=train_test_split(test_X,test_Y,test_size=0.2,random_state=temp)
       
        test_five_x.append(temp_test_x)
        test_five_y.append(temp_test_y)
 

final_accuracy=cross_validation(test_five_x,test_five_y,False)
print('five-fold test accuracy without smoothing =',round(sum(final_accuracy)/len(final_accuracy),2),'%')

```

    Accuracy for dataset 1  is = 77.1 %
    Accuracy for dataset 2  is = 78.3 %
    Accuracy for dataset 3  is = 77.28 %
    Accuracy for dataset 4  is = 77.78 %
    Accuracy for dataset 5  is = 76.94 %
    five-fold test accuracy without smoothing = 77.48 %
    

## five-fold cross validation for final accuracy by using test data set with smoothing


```python
final_accuracy_smooth=cross_validation(test_five_x,test_five_y,True)
print('five-fold test accuracy with smoothing =',round(sum(final_accuracy_smooth)/len(final_accuracy_smooth),2),'%')
```

    Accuracy for dataset 1  is = 79.72 %
    Accuracy for dataset 2  is = 80.52 %
    Accuracy for dataset 3  is = 79.56 %
    Accuracy for dataset 4  is = 79.62 %
    Accuracy for dataset 5  is = 79.26 %
    five-fold test accuracy with smoothing = 79.74 %
    


```python

```
