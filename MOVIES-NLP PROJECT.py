import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

insurance=pd.read_csv('/content/insurance (1) (1).csv')

insurance.head()

insurance.shape

insurance.info()

insurance.isnull().sum()

age_mean=insurance['age'].mean()

insurance['age'].fillna(age_mean,inplace=True)

insurance['children'].mode()

insurance['region'].unique()

insurance.describe()

sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance['age'])
plt.show()

plt.figure(figsize=(6,6))
sns.histplot(insurance['bmi'])
plt.title("BMI",fontsize=15,fontweight='bold')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='region',data=insurance)
plt.show()

insurance.head()

insurance['region'].value_counts()

encoder=LabelEncoder()

insurance['sex']=encoder.fit_transform(insurance['sex'])
insurance['smoker']=encoder.fit_transform(insurance['smoker'])
insurance['region']=encoder.fit_transform(insurance['region'])




insurance.head()

x=insurance.drop(columns=['bmi'],axis=1)
y=insurance['bmi']

x

y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

x_train_scaled

x_test.to_csv("insurance_test.csv",index=False)
from google.colab import files
files.download("insurance_test.csv")

print(x.shape,x_train.shape,x_test.shape)

reg=XGBRegressor()

reg.fit(x_train_scaled,y_train)

train_data_prediction=reg.predict(x_train_scaled)
r2_train=metrics.r2_score(y_train,train_data_prediction)
print("r squared value=", r2_train)

test_data_prediction=reg.predict(x_test_scaled)
r2_test=metrics.r2_score(y_test,test_data_prediction)
print(f"R Squared value: {r2_test}")

y_test

inputdata=(50,1,0,0,9617.66245)

inputarray=np.array(inputdata)

reshaped=inputarray.reshape(1,-1)
scale = scaler.transform(reshaped)

prediction=regressor.predict(scale)
print(prediction[0])

inputdata=(44,1,3,0,0,8891.1395)

inputarray=np.array(inputdata)

reshaped=inputarray.reshape(1,-1)
scale = scaler.transform(reshaped)

prediction=reg.predict(scale)
print(prediction[0])