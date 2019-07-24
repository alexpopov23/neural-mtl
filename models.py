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
    inputs = (keras.Input(shape=(max_seq_length,), name='Word_ids', dtype="int32"),
              keras.Input(shape=(max_seq_length,), name="Mask", dtype="float32"))
    emb_inputs = keras.layers.Embedding(len(embeddings),
                                        len(embeddings[0]),
                                        mask_zero=True,
                                        weights=[embeddings])(inputs[0])
    bilstm = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, return_sequences=True),
                                        name="BiLSTM",
                                        merge_mode="concat")(emb_inputs)
    dropout = keras.layers.Dropout(dropout, name="Dropout")(bilstm)
    unmasked = keras.layers.Multiply()([bilstm, tf.expand_dims(inputs[1], -1)])
    masked = keras.layers.Masking(mask_value=0.0)(unmasked)
    if method == "multitask":
        outputs1 = keras.layers.Dense(output_dim[0], activation='relu', name="Relu_classif")(masked)
        outputs2 = keras.layers.Dense(output_dim[1], activation='relu', name="Relu_embed")(masked)
        outputs = (keras.layers.Activation('softmax')(outputs1), outputs2)
    else:
        outputs = keras.layers.Dense(output_dim, activation='relu', name="Relu")(dropout)
        if method == "classification":
            outputs = keras.layers.Activation('softmax')(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def accuracy(predictions, possible_synsets, embeddings, true_preds, metric, method, metric2=None):
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
    choices, choices2 = [], []
    if method == "multitask":
        predictions = zip(predictions[0], predictions[1])
    for i, word in enumerate(predictions):
            # all_synsets = possible_synsets[i]
            if method == "classification":
                synset_ids = possible_synsets[i]
            elif method == "context_embedding":
                synset_embedding_ids = possible_synsets[i]
            elif method == "multitask":
                synset_ids = possible_synsets[0][i]
                synset_embedding_ids = possible_synsets[1][i]
            if method == "classification" or method == "multitask":
                activations = tf.gather(word[0], synset_ids)
                choices.append(tf.argmax(activations))
            if method == "context_embedding" or method == "multitask":
                possible_golds = tf.gather(embeddings, synset_embedding_ids)
                tiled_prediction = tf.tile(tf.reshape(word[1], [1, -1]), [len(synset_embedding_ids), 1])
                similarities = keras.losses.CosineSimilarity(reduction=losses_utils.ReductionV2.NONE)(possible_golds,
                                                                                                      tiled_prediction)
                if method == "multitask":
                    choices2.append(tf.argmax(similarities))
                else:
                    choices.append(tf.argmax(similarities))
    metric.update_state(true_preds, tf.stack(choices))
    result = metric.result().numpy()
    if method == "multitask" and metric2 is not None:
        metric2.update_state(true_preds, tf.stack(choices2))
        result2 = metric2.result().numpy()
    else:
        result2 = 0.0
    return result, result2

class MultitaskLoss():
    """Takes two loss functions for separate pathways in the neural net and returns the sum of their results
    """

    def __init__(self, loss1, loss2):
        self.loss1 = loss1
        self.loss2 = loss2

    def __call__(self, true_preds, outputs):
        loss_value1 = self.loss1(true_preds[0], outputs[0])
        loss_value2 = self.loss2(true_preds[1], outputs[1])
        return loss_value1 + loss_value2
