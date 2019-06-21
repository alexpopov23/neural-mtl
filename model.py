import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras.utils import losses_utils

class CosineSimilarityForAccuracy(keras.metrics.Metric):

    def __init__(self, name='cosine_similarity_accuracy', **kwargs):
        super(CosineSimilarityForAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_poss = y_true
        y_true = tf.unstack(y_true)
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)
        for i, gold_tensor in enumerate(y_true):
            max_similarity = -1.0
            for possible_tensor in y_poss:
                similarity = cosine_loss(possible_tensor, gold_tensor)
                if similarity > max_similarity:
                    selected = possible_tensor
                    max_similarity = similarity
            # if selected == y_true[]


        y_pred = tf.argmax(y_pred)
        values = tf.equal(tf.cast(y_true, 'int32'), tf.cast(y_pred, 'int32'))
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)

class WSDAccuracyCosineSimilarity(keras.layers.Layer):
    """
    algorithm for calculating the custom metric:
    - reshape outputs to [batch, seq, 1, 300]
    - tile outputs to [batch, seq, k, 300] where k is number of possible senses
    - compute cosine_similarity with reduction=(tf.python.keras.utils.)losses_utils.ReductionV2.NONE
    - get argmax at dim=2
    - compare with labels represented as integers (pass them as inputs?)
    """
    def accuracy(self, predictions, possible_synsets, max_synsets, true_preds, sample_weight):
        shape = predictions.shape.dims.insert(1)
        predictions = tf.reshape(predictions, shape)
        predictions = tf.tile(predictions, [1, 1, max_synsets, 1])
        similarities = keras.losses.cosine_similarity(possible_synsets,
                                                      predictions,
                                                      reduction=losses_utils.ReductionV2.NONE)
        chosen_synsets = tf.argmax(similarities)
        accuracy = keras.metrics.accuracy(true_preds, chosen_synsets, sample_weigth=sample_weight)
        return accuracy

    def call(self, inputs):
        outputs, golds, possibles = inputs
        self.add_metric(self.accuracy(inputs),
                        name='dictionary_based_accuracy',
                        aggregation='mean')
        return inputs  # Pass-through layer.

def get_model(method, embeddings1, output_dim, max_seq_length, n_hidden, dropout):
    """Creates the NN model

    Args:
        method: A string, the kind of disambiguation performed by the model (classification, context embedding, etc.)
        embeddings1: A Tensor with the word embeddings
        output_dim: An int, the size of the expected output
        max_seq_length: The maximum length of the data sequences (used in LSTM to save computational resources)
        n_hidden: An int, the size of the individual layers in the LSTMs
        dropout: A float, the probability of dropping the activity of a neuron (dropout)
    Return:
        model: tf.keras.Model()
    """
    inputs = keras.Input(shape=(max_seq_length,), name='Word_ids', dtype="int32")
    emb_inputs = tf.gather(embeddings1, inputs)
    bilstm = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, return_sequences=True),
                                        name="BiLSTM",
                                        merge_mode="concat")(emb_inputs)
    dropout = keras.layers.Dropout(dropout, name="Dropout")(bilstm)
    outputs = keras.layers.Dense(output_dim, activation='relu', name="Relu")(dropout)
    if method == "classification":
        outputs = keras.layers.Dense(output_dim, activation='softmax', name="Softmax")(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def accuracy(predictions, possible_synsets, embeddings, true_preds):
    choices = []
    for i, sent in enumerate(predictions):
        for j, word in enumerate(sent):
            all_synsets = possible_synsets[i][j]
            tiled_prediction = tf.tile(word, [len(all_synsets), 1])
            similarities = keras.losses.cosine_similarity(possible_synsets,
                                                          tiled_prediction,
                                                          reduction=losses_utils.ReductionV2.NONE)
            choices.append(tf.argmax(similarities))
    accuracy = keras.metrics.accuracy(true_preds, choices)
    return accuracy


