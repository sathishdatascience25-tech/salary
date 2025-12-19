
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # df=pd.read_csv("C:/Users/ADMIN/Downloads/Salary_dataset.csv")
# print(df.head())
# print(df.columns)
# sns.scatterplot(x="YearsExperience",y="Salary",data=df)
# plt.title("years vs salary")
# plt.show()

# sns.lineplot(x="YearsExperience",y="Salary",data=df)
# plt.title("years vs salary")
# plt.show()

# print(df.info())

# print(df.describe())


# sns.histplot(df["YearsExperience"],kde=True)
# plt.show()

# print(df.isnull().sum())
# print(df[df.duplicated()])

# print(df.head())
# df["Salary"]=df["Salary"].astype(int)


def salary_level(years):
    if years <3:
        return 0
    elif years <5:
        return 1
    elif years <8:
        return 2
    else :
        return 3
# df["exp_level"]   =df["YearsExperience"].apply(salary_level)
# print(df.tail())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

X=df.drop(columns=["Unnamed: 0","Salary"])
y=df["Salary"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

std=StandardScaler()
X_train=std.fit_transform(X_train)
X_test=std.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
model_lr=LinearRegression()
model_lr.fit(X_train,y_train)

y_pred_lr=model_lr.predict(X_test)

model_lasso=Lasso()
model_lasso.fit(X_train,y_train)
y_pred_lasso=model_lasso.predict(X_test)

model_ridge=Ridge()
model_ridge.fit(X_train,y_train)
y_pred_ridge=model_ridge.predict(X_test)


def evaluate(model,y_test,y_pred):
    print(model)
    print("r2_score",r2_score(y_test,y_pred))
    print("mean_squared_error",mean_squared_error(y_test,y_pred))
    print("mean_absolute_error",mean_absolute_error(y_test,y_pred))


evaluate("linear_regression",y_test,y_pred_lr)
evaluate("lasso",y_test,y_pred_lasso)
evaluate("ridge",y_test,y_pred_ridge)

error=y_test-y_pred_lr
print(error)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

poly_model=LinearRegression()
poly_model.fit(X_train_poly,y_train)
y_pred_poly=poly_model.predict(X_test_poly)

evaluate("poly",y_test,y_pred_poly)
import numpy as np
import joblib
joblib.dump(model_ridge,"model.pkl")
model=joblib.load("model.pkl")
import streamlit as st
joblib.dump(std,"scaler.pkl")
scaler=joblib.load("scaler.pkl")

st.title("salary prediction")
st.write("predict using  a ridge regression ")
exp=st.number_input("enter a years Exp")
exp_level=st.number_input("exp_leve")
if st.button("predict"):
    input_=np.array([[exp,exp_level]])
    in_scaler=scaler.transform(input_)

    salary=model.predict(in_scaler)
    st.success(salary[0])    

