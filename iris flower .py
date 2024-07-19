import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
%matplotlib inline
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
df.head(150)
df.describe()
sns.pairplot(df,hue="class")
data=df.values
x=data[:,0:4]
y=data[:,4]
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
print(y_test)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
from sklearn.metrics import classification_report, accuracy_score
x_pred=knn.predict(x_test)
print("accuracy:",accuracy_score(y_test,x_pred)*100)
for i in range(len(x_pred)):
    print(y_test[i],x_pred[i])
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
x_pred2=LR.predict(x_test)
print("accuracy:",accuracy_score(y_test,x_pred2)*100)
for i in range(len(x_pred2)):
    print(y_test[i],x_pred2[i])
print(classification_report(y_test,y_pred))
x_new=np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[0.1,2,0,1]])
prediction=knn.predict(x_new)
print("prediction of species:{}".format(prediction))