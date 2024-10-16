# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SADHANA SHREE B
RegisterNumber: 212223230177 
*/
import pandas as pd
data=pd.read_csv("Salary (2).csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
print("mse : ",mse)
r2=metrics.r2_score(y_test,y_pred)
print("r2 : ",r2)
dt.predict([[5,6]])
```

## Output:

![Screenshot 2024-10-16 114600](https://github.com/user-attachments/assets/b55acedb-ba21-408b-8a83-bded62f8935d)

![Screenshot 2024-10-16 114604](https://github.com/user-attachments/assets/f76f6b13-ef00-4e28-a2f2-7b47b90538a7)

![Screenshot 2024-10-16 114608](https://github.com/user-attachments/assets/d89a6714-765b-44a4-b59d-84bd762758de)

![Screenshot 2024-10-16 114614](https://github.com/user-attachments/assets/87b19a27-e805-46fa-911f-19a4790a8b85)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
