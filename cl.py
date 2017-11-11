##import sys
##reload(sys)
##sys.setdefaultencoding("utf-8")

import random
#from nltk.corpus import movie_reviews
from textblob.classifiers import DecisionTreeClassifier
#NaiveBayesClassifier
random.seed(1)
import pandas as pd
data= pd.read_csv('boo.csv')
##data=open('TRAIN_SMS.csv','w')
#print data
##col_list = list(data)
##
##col_list[0], col_list[1] = col_list[1], col_list[0]

data=data[['Message','Label']]
##print data
##
##print data
data=data.values
####print data
ol=[]
for d in data:
    ol.append(d.tolist())
##print ol
train=ol[:3000]
test=ol[29900:]


##
##train = [
##    ('I love this sandwich.', 'pos'),
##    ('This is an amazing place!', 'pos'),
##    ('I feel very good about these beers.', 'pos'),
##    ('This is my best work.', 'pos'),
##    ("What an awesome view", 'pos'),
##    ('I do not like this restaurant', 'neg'),
##    ('I am tired of this stuff.', 'neg'),
##    ("I can't deal with this", 'neg'),
##    ('He is my sworn enemy!', 'neg'),
##    ('My boss is horrible.', 'neg')
##]
##test = [
##    ('The beer was good.', 'pos'),
##    ('I do not enjoy my job', 'neg'),
##    ("I ain't feeling dandy today.", 'neg'),
##    ("I feel amazing!", 'pos'),
##    ('Gary is a friend of mine.', 'pos'),
##    ("I can't believe I'm doing this.", 'neg')
##]
##
#cl = NaiveBayesClassifier(train)
#cl = DecisionTreeClassifier(train)



# Grab some movie review data
##reviews = [(list(movie_reviews.words(fileid)), category)
##              for category in movie_reviews.categories()
##              for fileid in movie_reviews.fileids(category)]
##random.shuffle(reviews)
##new_train, new_test = reviews[0:100], reviews[101:200]
##accuracy = cl.accuracy(test)
##print(accuracy)
### Update the classifier with the new training data
##cl.update(ol[25000:28000])
##accuracy = cl.accuracy(test)
##print(accuracy)
###cl.update(ol[:50])
##cl.update(ol[15000:20000])
##accuracy = cl.accuracy(test)
##print(accuracy)
##
### Compute accuracy
import _pickle as cPickle
##
### save the classifier
##with open('windows_class_.pkl', 'wb') as fid:
##    cPickle.dump(cl, fid)    

# load it again
with open('windows_class_.pkl', 'rb') as fid:
    cl = cPickle.load(fid)






accuracy = cl.accuracy(test)
print("Accuracy: {0}".format(accuracy))



res= pd.read_csv('foo.csv')


res=res.values
#print res
pl=[]
for r in res:
    pl.append(r[1])
##print pl
##pred=cl.classify(pl)
##print(pred)
o=1
for pli in pl:
    pred=cl.classify(pli)
    print("%s,%s"%(str(o),pred))
    o=o+1

# Show 5 most informative features
#cl.show_informative_features(5)
