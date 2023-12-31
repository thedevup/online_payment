import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Loading the csv file
df = pd.read_csv('../fin_data.csv')

# Exploring the dataset
print(df.head())

# Check the shape of the dataset
print(df.shape)

# Get the statistical summary
print(df.describe())

# Show the comulmnsand data types
print(df.info())

print('The number of missing values: \n' , df.isna().sum())

# One hot encoding the payment type
df2 = pd.get_dummies(df.type, prefix='type')
print(df2.head())

# Append the new features to the original dataframe
df = df.join(df2[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']])

# Drop the columns we are not interested in
df.drop(['nameOrig', 'nameDest', 'type', 'isFlaggedFraud'], axis=1, inplace=True)

# The shape of the dataset after dropping the unecessary features
print(df.shape)

# Check if the target variable is balanced
print(df['isFraud'].value_counts())

# Matrix of histograms
df.hist(figsize=(12, 8))
plt.tight_layout()  # ensures that the plots do not overlap
plt.show()

# Visual correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation matrix')
plt.show()

# Loadthe dependent and undependent varialbles
X = df.drop('isFraud',axis=1)
Y = df['isFraud']

# Splitting the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=6)

# Train the classifier
Y_pred_knn = knn.fit(X_train, Y_train).predict(X_test)
print(Y_pred_knn)

# Create the NB model
nb = GaussianNB()

# Train the classifier
Y_pred_nb = nb.fit(X_train, Y_train).predict(X_test)

print(Y_pred_nb)

# Calculate the accyracy for knn
print(classification_report(Y_test,Y_pred_knn))

# knn accuracy score
print("knn accuracy score: ", accuracy_score(Y_test,Y_pred_knn))

# Calculate the accyracy for NB
print(classification_report(Y_test,Y_pred_nb))

# NB accuracy score
print("nb accuracy score: ", accuracy_score(Y_test,Y_pred_nb))
