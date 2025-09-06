import pandas as pd
import numpy as np
from random_forest import RandomForest
from utils import encode_dataframe, encode_series
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/cars.csv'))
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
data = pd.read_csv(DATA_PATH, names=columns)
features = data.drop('label', axis=1)
target = data['label']

features_encoded, _ = encode_dataframe(features)
target_encoded, target_mapping = encode_series(target)

features_encoded = np.array(features_encoded)
target_encoded = np.array(target_encoded)

X_train, X_test, y_train, y_test = train_test_split(features_encoded, target_encoded, test_size=0.2, random_state=42)

# Train RandomForest
rf = RandomForest(n_trees=10)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Metrics
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nSample predictions:')
for i in range(10):
    print(f'Predicted: {y_pred[i]}, Actual: {y_test[i]}')
