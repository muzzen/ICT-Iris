# Import Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Import dataset
df = pd.read_excel('iris.xls')

# Define features and target
x = df.iloc[:,0:-1]
y = df['Classification']

# Split the dataset
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size= 0.1
)

# Creat the model
classifier = RandomForestClassifier()
classifier.fit(train_x.values, train_y)

# Picle the model
pickle.dump(classifier,open('classifier.pkl','wb'))