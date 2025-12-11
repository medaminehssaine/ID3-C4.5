"""data loaders for c4.5 (with continuous features)"""


def load_iris():
    """
    classic iris dataset with continuous features
    good for testing c4.5's threshold handling
    """
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # subset of iris data (numeric)
    data = [
        (5.1, 3.5, 1.4, 0.2, "setosa"),
        (4.9, 3.0, 1.4, 0.2, "setosa"),
        (4.7, 3.2, 1.3, 0.2, "setosa"),
        (4.6, 3.1, 1.5, 0.2, "setosa"),
        (5.0, 3.6, 1.4, 0.2, "setosa"),
        (5.4, 3.9, 1.7, 0.4, "setosa"),
        (4.6, 3.4, 1.4, 0.3, "setosa"),
        (5.0, 3.4, 1.5, 0.2, "setosa"),
        (7.0, 3.2, 4.7, 1.4, "versicolor"),
        (6.4, 3.2, 4.5, 1.5, "versicolor"),
        (6.9, 3.1, 4.9, 1.5, "versicolor"),
        (5.5, 2.3, 4.0, 1.3, "versicolor"),
        (6.5, 2.8, 4.6, 1.5, "versicolor"),
        (5.7, 2.8, 4.5, 1.3, "versicolor"),
        (6.3, 3.3, 4.7, 1.6, "versicolor"),
        (4.9, 2.4, 3.3, 1.0, "versicolor"),
        (6.3, 3.3, 6.0, 2.5, "virginica"),
        (5.8, 2.7, 5.1, 1.9, "virginica"),
        (7.1, 3.0, 5.9, 2.1, "virginica"),
        (6.3, 2.9, 5.6, 1.8, "virginica"),
        (6.5, 3.0, 5.8, 2.2, "virginica"),
        (7.6, 3.0, 6.6, 2.1, "virginica"),
        (4.9, 2.5, 4.5, 1.7, "virginica"),
        (7.3, 2.9, 6.3, 1.8, "virginica"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names


def load_wine_sample():
    """
    wine quality sample (continuous features)
    """
    feature_names = ["alcohol", "malic_acid", "ash", "alcalinity"]
    
    data = [
        (14.23, 1.71, 2.43, 15.6, "class_1"),
        (13.20, 1.78, 2.14, 11.2, "class_1"),
        (13.16, 2.36, 2.67, 18.6, "class_1"),
        (14.37, 1.95, 2.50, 16.8, "class_1"),
        (13.24, 2.59, 2.87, 21.0, "class_1"),
        (12.37, 0.94, 1.36, 10.6, "class_2"),
        (12.33, 1.10, 2.28, 16.0, "class_2"),
        (12.64, 1.36, 2.02, 16.8, "class_2"),
        (13.67, 1.25, 1.92, 18.0, "class_2"),
        (12.37, 1.13, 2.16, 19.0, "class_2"),
        (12.17, 1.45, 2.53, 19.0, "class_3"),
        (12.37, 1.21, 2.56, 18.1, "class_3"),
        (13.11, 1.01, 1.70, 15.0, "class_3"),
        (12.37, 1.17, 1.92, 19.6, "class_3"),
        (13.34, 0.94, 2.36, 17.0, "class_3"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names


def load_golf():
    """
    golf/weather dataset - mix of categorical and "continuous-ish"
    good for showing c4.5 handling both types
    """
    feature_names = ["outlook", "temperature", "humidity", "windy"]
    
    # temperature and humidity as numbers
    data = [
        ("sunny", 85, 85, "false", "no"),
        ("sunny", 80, 90, "true", "no"),
        ("overcast", 83, 78, "false", "yes"),
        ("rain", 70, 96, "false", "yes"),
        ("rain", 68, 80, "false", "yes"),
        ("rain", 65, 70, "true", "no"),
        ("overcast", 64, 65, "true", "yes"),
        ("sunny", 72, 95, "false", "no"),
        ("sunny", 69, 70, "false", "yes"),
        ("rain", 75, 80, "false", "yes"),
        ("sunny", 75, 70, "true", "yes"),
        ("overcast", 72, 90, "true", "yes"),
        ("overcast", 81, 75, "false", "yes"),
        ("rain", 71, 80, "true", "no"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names


def load_with_missing():
    """
    dataset with missing values (None) for testing
    """
    feature_names = ["size", "color", "shape"]
    
    data = [
        ("large", "red", "round", "yes"),
        ("small", "red", "round", "yes"),
        ("large", None, "square", "no"),    # missing color
        ("small", "blue", None, "no"),      # missing shape
        ("large", "blue", "round", "yes"),
        (None, "red", "square", "no"),      # missing size
        ("small", "blue", "square", "no"),
        ("large", "red", "round", "yes"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names
