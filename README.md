# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Developing a neural network regression model entails a structured process, encompassing phases such as data acquisition, preprocessing, feature selection, model architecture determination, training, hyperparameter optimization, performance evaluation, and deployment, followed by ongoing monitoring for refinement.

## Neural Network Model

![image](https://github.com/user-attachments/assets/f6252a99-ebc6-40fc-a88f-17963b3c0d31)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SHARMILA A
### Register Number: 212221230094
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open('e1').sheet1
data=worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype(float)
dataset1.head()
x=dataset1.values
y=dataset1.values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
x_train=Scaler.transform(x_train)
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x_train,y_train,epochs=20)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
ai_brain.evaluate(x_test,y_test)
X_n1 = [[3,5]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)





```
## Dataset Information

1![image](https://github.com/user-attachments/assets/51ae34b0-b41f-4629-80df-a5803510f14b)


## OUTPUT
![image](https://github.com/user-attachments/assets/d0cd22fd-3007-41d3-b9ed-120b3de2b9a2)


### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/29761b03-6478-44cb-9397-e7b35cf65dd4)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/1a5cf049-574f-4681-83bb-fe93a4875617)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/81b24923-5d19-4799-9a7c-0b017a8da912)


## RESULT

Thus a neural network regression model for the given dataset has been developed.
