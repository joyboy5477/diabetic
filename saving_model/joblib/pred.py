import joblib

#load model
lr = joblib.load("diabetes_80.pkl")

# prediction
pred = lr.predict([[6,3,1,8,9,2,4,8]])

""" if you want to save the above sample also you can do that
f = open("newdata.csv","a+)
f.read([[6,3,1,8,9,2,4,8]])
"""
if pred[0] == 0:
    print("person is not diabetic")
else:
    print("person is diabetics")