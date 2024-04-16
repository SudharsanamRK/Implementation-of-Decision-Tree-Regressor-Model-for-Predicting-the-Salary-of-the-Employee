# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packagesprint the present data
2. print the present data
3. print the null value
4. using decisiontreeRegressor, find the predicted values,mse,r2
5. print the result
## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sudharsanam R K
RegisterNumber:  212222040163
```
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load the dataset
data = pd.read_csv("/content/Salary_EX7.csv")
data.head()

# Get information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Encode the 'Position' column
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

# Select features (independent variables) and target variable (dependent variable)
x = data[["Position", "Level"]]
y = data["Salary"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
mse

# Calculate R-squared score
r2 = metrics.r2_score(y_test, y_pred)
r2

# Predict salary for given Position and Level
dt.predict([[5, 6]])

# Plot the decision tree
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
## Dataset
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115523484/5be21b60-9941-4a03-a98c-2e8af748db1a)

##  Mean Squared Error
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115523484/6652e9b0-9825-4a4d-b0c6-27337afd0979)

##  R-squared score
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115523484/9b3fa3b1-a561-4bda-adf2-30cf388a1a04)

## Predicted Salary
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115523484/b25ea4c7-8358-4941-83fd-97e67dd7e212)

## Decision tree
![image](https://github.com/SudharsanamRK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/115523484/94309109-39ba-4071-b01d-43822a22612c)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
