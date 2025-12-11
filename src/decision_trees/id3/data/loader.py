"""dataset loading utilities"""
import csv
import os


def load_csv(filepath, has_header=True, label_col=-1):
    """
    load dataset from csv file
    
    returns: X (features), y (labels), feature_names
    """
    X = []
    y = []
    feature_names = None
    
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        if has_header:
            header = next(reader)
            # all columns except label
            if label_col == -1:
                feature_names = header[:-1]
            else:
                feature_names = [h for i, h in enumerate(header) if i != label_col]
        
        for row in reader:
            if not row:
                continue
            
            if label_col == -1:
                X.append(tuple(row[:-1]))
                y.append(row[-1])
            else:
                X.append(tuple(v for i, v in enumerate(row) if i != label_col))
                y.append(row[label_col])
    
    return X, y, feature_names


def load_play_tennis():
    """
    classic play tennis dataset
    quinlan's original example for id3
    """
    feature_names = ["outlook", "temperature", "humidity", "wind"]
    
    # outlook, temp, humidity, wind -> play
    data = [
        ("sunny", "hot", "high", "weak", "no"),
        ("sunny", "hot", "high", "strong", "no"),
        ("overcast", "hot", "high", "weak", "yes"),
        ("rain", "mild", "high", "weak", "yes"),
        ("rain", "cool", "normal", "weak", "yes"),
        ("rain", "cool", "normal", "strong", "no"),
        ("overcast", "cool", "normal", "strong", "yes"),
        ("sunny", "mild", "high", "weak", "no"),
        ("sunny", "cool", "normal", "weak", "yes"),
        ("rain", "mild", "normal", "weak", "yes"),
        ("sunny", "mild", "normal", "strong", "yes"),
        ("overcast", "mild", "high", "strong", "yes"),
        ("overcast", "hot", "normal", "weak", "yes"),
        ("rain", "mild", "high", "strong", "no"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names


def load_mushroom_sample():
    """
    sample mushroom dataset (subset)
    demonstrates id3 on a slightly larger dataset
    """
    feature_names = ["cap_shape", "cap_color", "odor", "gill_size", "stalk_shape"]
    
    # simplified mushroom data
    data = [
        ("convex", "brown", "none", "narrow", "enlarging", "edible"),
        ("convex", "yellow", "almond", "broad", "enlarging", "edible"),
        ("bell", "white", "none", "broad", "tapering", "edible"),
        ("convex", "white", "foul", "narrow", "enlarging", "poisonous"),
        ("convex", "brown", "foul", "narrow", "tapering", "poisonous"),
        ("bell", "brown", "none", "broad", "enlarging", "edible"),
        ("flat", "yellow", "none", "broad", "tapering", "edible"),
        ("flat", "white", "foul", "narrow", "tapering", "poisonous"),
        ("convex", "brown", "almond", "broad", "enlarging", "edible"),
        ("bell", "yellow", "foul", "narrow", "enlarging", "poisonous"),
        ("flat", "brown", "none", "broad", "tapering", "edible"),
        ("convex", "white", "none", "broad", "enlarging", "edible"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names


def load_iris_categorical():
    """
    iris dataset with discretized features
    for testing on a classic ml dataset
    """
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # pre-discretized iris samples
    data = [
        ("short", "wide", "short", "narrow", "setosa"),
        ("short", "wide", "short", "narrow", "setosa"),
        ("medium", "medium", "short", "narrow", "setosa"),
        ("short", "narrow", "short", "narrow", "setosa"),
        ("medium", "wide", "short", "narrow", "setosa"),
        ("medium", "medium", "medium", "medium", "versicolor"),
        ("long", "narrow", "medium", "medium", "versicolor"),
        ("medium", "medium", "medium", "medium", "versicolor"),
        ("medium", "narrow", "medium", "medium", "versicolor"),
        ("long", "medium", "medium", "medium", "versicolor"),
        ("long", "medium", "long", "wide", "virginica"),
        ("long", "narrow", "long", "medium", "virginica"),
        ("long", "medium", "long", "wide", "virginica"),
        ("medium", "narrow", "long", "wide", "virginica"),
        ("long", "wide", "long", "wide", "virginica"),
    ]
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y, feature_names
