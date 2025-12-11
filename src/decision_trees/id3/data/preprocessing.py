"""data preprocessing utilities"""
from collections import defaultdict


class LabelEncoder:
    """encode categorical labels as integers"""
    
    def __init__(self):
        self.classes_ = []
        self._mapping = {}
        self._inverse = {}
    
    def fit(self, y):
        """learn the label mapping"""
        self.classes_ = sorted(set(y))
        self._mapping = {label: i for i, label in enumerate(self.classes_)}
        self._inverse = {i: label for label, i in self._mapping.items()}
        return self
    
    def transform(self, y):
        """transform labels to integers"""
        return [self._mapping[label] for label in y]
    
    def fit_transform(self, y):
        """fit and transform in one step"""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y_encoded):
        """convert back to original labels"""
        return [self._inverse[i] for i in y_encoded]


def encode_labels(y):
    """quick label encoding, returns encoded labels and encoder"""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return encoded, encoder


def discretize(values, bins=3, labels=None):
    """
    discretize continuous values into bins
    
    values: list of numeric values
    bins: number of bins or list of bin edges
    labels: optional labels for bins (e.g., ['low', 'medium', 'high'])
    """
    if isinstance(bins, int):
        # create equal-width bins
        min_val = min(values)
        max_val = max(values)
        width = (max_val - min_val) / bins
        bin_edges = [min_val + i * width for i in range(bins + 1)]
        bin_edges[-1] = max_val + 0.001  # include max value
    else:
        bin_edges = bins
    
    n_bins = len(bin_edges) - 1
    
    if labels is None:
        labels = [f"bin_{i}" for i in range(n_bins)]
    
    result = []
    for v in values:
        for i in range(n_bins):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                result.append(labels[i])
                break
        else:
            result.append(labels[-1])
    
    return result


def discretize_column(X, col_idx, bins=3, labels=None):
    """discretize a specific column in the dataset"""
    values = [sample[col_idx] for sample in X]
    discretized = discretize(values, bins, labels)
    
    # rebuild X with discretized column
    new_X = []
    for i, sample in enumerate(X):
        new_sample = list(sample)
        new_sample[col_idx] = discretized[i]
        new_X.append(tuple(new_sample))
    
    return new_X


def one_hot_encode(X, col_idx):
    """
    one-hot encode a categorical column
    returns new X with one column per category
    """
    # get unique values
    values = set(sample[col_idx] for sample in X)
    value_list = sorted(values)
    
    new_X = []
    for sample in X:
        old_val = sample[col_idx]
        # remove original column
        new_sample = list(sample[:col_idx]) + list(sample[col_idx + 1:])
        # add one-hot columns
        for v in value_list:
            new_sample.append("1" if old_val == v else "0")
        new_X.append(tuple(new_sample))
    
    return new_X, value_list
