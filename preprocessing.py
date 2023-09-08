# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Loading the csv file
df = pd.read_csv('../fin_data.csv')

# Exploring the fin_data
print(df.head)

# Show the comulmnsand data types
print(df.info())

print('The number of missing values: \n' , df.isna().sum())

# Drop the columns we are not interested in
df.drop(['nameOrig', 'nameDest', "type"], axis=1, inplace=True)

# Correlation matrix
print(df.corr())

# Loadthe dependent and undependent varialbles
X = df.drop('isFraud',axis=1)
Y = df['isFraud']

# Splitting the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=1)

# Train the KNN model

# Train the NB model
