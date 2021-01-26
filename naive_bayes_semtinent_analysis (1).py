import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = 'smsspamcollection/SMSSpamCollection.csv'

df=pd.read_csv(file_path,sep='\t',names=['Status','Message'])

df.loc[df["Status"]=='ham',"Status",]=1
df.loc[df["Status"]=='spam',"Status",]=0

df_x=df["Message"]
df_y=df["Status"]


cv = TfidfVectorizer(min_df=1,stop_words='english')



"""from pandas import DataFrame
def create_document_term_matrix(message_list, vectorizer):
    doc_term_matrix = vectorizer.fit_transform(message_list)
    return DataFrame(doc_term_matrix.toarray(), 
                     columns=vectorizer.get_feature_names())

msg_1 = ["navie bays algorihm",
        "I am snehali teaching you navie bays "]

count_vect = CountVectorizer()
create_document_term_matrix(msg_1, count_vect)

msg_2=['snehali is my name',
       'snehali likes python programming']

tfidf_vect = TfidfVectorizer()
create_document_term_matrix(msg_2, tfidf_vect)"""



x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


x_traincv=cv.fit_transform(x_train)

a=x_traincv.toarray()

a[0]

cv.inverse_transform(a[0])
x_train.iloc[0]

x_testcv=cv.transform(x_test)
#x_testcv.toarray()

mnb = MultinomialNB()
y_train=y_train.astype('int')
mnb.fit(x_traincv,y_train)
predictions=mnb.predict(x_testcv)
predictions

actual = np.array(y_test)
count=0
for i in range (len(predictions)):
    if predictions[i]==actual[i]:
        count=count+1
        
len(predictions)
len(y_test)
        
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
print(confusion_matrix(predictions, y_test.astype(int)))
print(classification_report(y_test.astype(int), mnb.predict(x_testcv), digits=4))
score = mnb.score(x_traincv,y_train)



from sklearn.metrics import roc_curve, auc
y_score = mnb.predict_proba(x_testcv)[:,1]
fpr, tpr, _ = roc_curve(y_test.astype(int), y_score)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.title('ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print('Area under curve (AUC): ', auc(fpr,tpr))
