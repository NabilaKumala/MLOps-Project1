
import tensorflow_transform as tft
import tensorflow as tf

LABEL_KEY = "Suicide"
FEATURE_KEY = "Tweet"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
