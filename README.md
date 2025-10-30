# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. **Data Loading and Preprocessing**  
   - Read the dataset (`spam.csv`), detect file encoding, and handle missing values.  
   - Extract features (`v2`) as message text and labels (`v1`) as spam/ham.

2. **Data Splitting**  
   - Split the dataset into training and testing sets using `train_test_split` (e.g., 80% training, 20% testing).

3. **Feature Extraction**  
   - Convert text data into numerical features using `CountVectorizer`.  
   - Fit on the training data and transform both training and test sets.

4. **Model Training and Evaluation**  
   - Train the Support Vector Machine (`SVC`) model on the training data.  
   - Predict on the test data and evaluate performance using accuracy score, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: CHARUKESH S
RegisterNumber:  21224230044
*/
```
```python
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result


import pandas as pd
df=pd.read_csv("spam.csv",encoding='Windows-1252')
df.head()

df.info()

df.isnull().sum()

x=df["v2"].values
y=df["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train

x_test

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)

x_train

x_test

from sklearn.svm import SVC
svc = SVC()                 
svc.fit(x_train, y_train)   
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("NAME: CHARUKESH S")
print("REG NO: 212224230044")
accuracy

from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_test,y_pred)
print("NAME: CHARUKESH S")
print("REG NO: 212224230044")
conf

class_report=metrics.classification_report(y_test,y_pred)
print("NAME: CHARUKESH S")
print("REG NO: 212224230044")
class_report
```

## Output:
<img width="976" height="55" alt="image" src="https://github.com/user-attachments/assets/ac35b939-d90e-4353-a72f-2216bf29dad3" />
<img width="953" height="228" alt="image" src="https://github.com/user-attachments/assets/542c378f-2b26-4466-958d-d6edd4a5a0b8" />
<img width="987" height="286" alt="image" src="https://github.com/user-attachments/assets/f4460f9e-5c75-44ff-9d02-1ecb27a08b35" />
<img width="997" height="149" alt="image" src="https://github.com/user-attachments/assets/c1e21e7e-45d4-47c2-b638-7def4844969b" />
<img width="1251" height="215" alt="image" src="https://github.com/user-attachments/assets/dedb50ff-860d-48ae-9222-b9a4181bbdfd" />
<img width="1245" height="254" alt="image" src="https://github.com/user-attachments/assets/30663684-3b0a-4add-b8a1-42d737a6cb80" />
<img width="992" height="72" alt="image" src="https://github.com/user-attachments/assets/c7be8817-b0ee-4632-aeb2-62addc776fa8" />
<img width="974" height="67" alt="image" src="https://github.com/user-attachments/assets/7d772eb7-6015-4d24-9a36-6ed57ab1d4d0" />
<img width="1091" height="47" alt="image" src="https://github.com/user-attachments/assets/93cf421e-e4cc-4ee5-89e5-be3e10c7b2c8" />
<img width="1084" height="114" alt="image" src="https://github.com/user-attachments/assets/4982db96-040b-434b-9ff3-f65b3bf68b4a" />
<img width="1087" height="142" alt="image" src="https://github.com/user-attachments/assets/f931c8ee-b66a-4923-ad6b-051ecb370340" />
<img width="1385" height="167" alt="image" src="https://github.com/user-attachments/assets/3b8ef6c8-8aa0-4022-a7d6-770abd75c986" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
