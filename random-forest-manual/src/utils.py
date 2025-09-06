import pandas as pd

def encode_dataframe(df):
    """
    Encode all columns in a DataFrame to integers.
    Returns encoded DataFrame and mapping dicts for each column.
    """
    encoders = {}
    df_encoded = df.copy()
    for col in df.columns:
        df_encoded[col], encoders[col] = pd.factorize(df[col])
    return df_encoded, encoders

def encode_series(series):
    """
    Encode a pandas Series to integers.
    Returns encoded Series and mapping dict.
    """
    encoded, mapping = pd.factorize(series)
    return encoded, mapping
def calculate_entropy(y):
    from collections import Counter
    import math

    total = len(y)
    if total == 0:
        return 0

    counts = Counter(y)
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def split_dataset(X, y, feature_index, threshold):
    left_indices = [i for i in range(len(X)) if X[i][feature_index] <= threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_index] > threshold]

    X_left = [X[i] for i in left_indices]
    y_left = [y[i] for i in left_indices]
    X_right = [X[i] for i in right_indices]
    y_right = [y[i] for i in right_indices]

    return (X_left, y_left), (X_right, y_right)

def majority_vote(y):
    from collections import Counter
    return Counter(y).most_common(1)[0][0]

def calculate_accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) if len(y_true) > 0 else 0

def get_random_subset(X, y, sample_size):
    import random
    indices = random.sample(range(len(X)), sample_size)
    return [X[i] for i in indices], [y[i] for i in indices]