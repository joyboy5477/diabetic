import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
import joblib # you have to import this!

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ["preg","plas","pres","skin","test","mass","pedi","age","class"]
data  = pd.read_csv(url,names = names)

#print(data.head())

df = data.copy()

#print(data.shape)

array = df.values

# take x and y
x = array[:,0:8]
y = array[:,8]

#split
xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.2,random_state=42)

#train the model
lr = LogisticRegression(max_iter=1000)# do this max_iter otherwise you get error of limit reached
lr.fit(xtrain,ytrain)

#accuracy of test
result = lr.score(xtest,ytest)
#print(result)

# model saving
filename = "diabetes_80.pkl" # you can also use .sav #pkl - pickle
joblib.dump(lr,filename,)
""" what we did above ? 
1. you are saving a trained logistic regression model using the Python pickle module
2. The .pkl extension indicates that you are using the pickle format to store the model.
3. you use the pickle.dump() function to serialize and save the trained logistic regression model. 
   The function takes two arguments: the model object (lr) and an open 
   file object (open(filename, "wb")) that represents the file where the model will be saved.
 """
