import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras.utils import losses_utils

def get_model(method, embeddings, output_dim, max_seq_length, n_hidden, dropout):
    """Creates the NN model

    Args:
        method: A string, the kind of disambiguation performed by the model (classification, context embedding, etc.)
        embeddings: A Tensor with the word embeddings
        output_dim: An int, the size of the expected output
        max_seq_length: The maximum length of the data sequences (used in LSTM to save computational resources)
        n_hidden: An int, the size of the individual layers in the LSTMs
        dropout: A float, the probability of dropping the activity of a neuron (dropout)
    Return:
        models: tf.keras.Model()
    """
    inputs = keras.Input(shape=(max_seq_length,), name='Word_ids', dtype="int32")
    emb_inputs = tf.gather(embeddings, inputs, name="Embed_inputs")
    bilstm = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, return_sequences=True),
                                        name="BiLSTM",
                                        merge_mode="concat")(emb_inputs)
    dropout = keras.layers.Dropout(dropout, name="Dropout")(bilstm)
    outputs = keras.layers.Dense(output_dim, activation='relu', name="Relu")(dropout)
    if method == "classification":
        outputs = keras.layers.Dense(output_dim, activation='softmax', name="Softmax")(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def accuracy(predictions, possible_synsets, embeddings, true_preds, metric):
    """Calculates accuracy of a model run

    Args:
        predictions: A list of tensors; vector predictions by the network, per word found in the sense lexicon
        possible_synsets: Lists with the possible synset IDs for each word is being disambiguated
        embeddings: A tensor; the embeddings from which to retrieve the possible gold synset vectors
        true_preds: A list of ints; the sequence number of the correct synsets per word (as they appear in the lexicon)
        metric: A keras.metric object; to be updated in each call to the function
    Returns:
        A scalar; the accuracy of the run.
    """
    choices = []
    for i, word in enumerate(predictions):
            all_synsets = possible_synsets[i]
            possible_golds = tf.gather(embeddings, all_synsets)
            tiled_prediction = tf.tile(tf.reshape(word, [1, -1]), [len(all_synsets), 1])
            similarities = keras.losses.CosineSimilarity(reduction=losses_utils.ReductionV2.NONE)(possible_golds,
                                                                                                  tiled_prediction)
            choices.append(tf.argmax(similarities))
    metric.update_state(true_preds, tf.stack(choices))
    return metric.result().numpy()


