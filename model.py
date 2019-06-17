import tensorflow as tf

from tensorflow.python import keras

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

class MetricLoggingLayer(keras.layers.Layer):

    def call(self, inputs):
        outputs, golds, possibles = inputs

        value = []
        self.add_metric(value,
                        name='dictionary_based_accuracy',
                        aggregation='mean')
        return inputs  # Pass-through layer.

def get_model(method, embeddings1, embeddings2, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, max_seq_length,
              n_hidden, dropout):
    """Creates the NN model

    Args:
        method: A string, the kind of disambiguation performed by the model (classification, context embedding, etc.)
        output_dim: An int, this is the size of the output layer, on which classification/regression is computed
        vocab_size1: An int, the number of words in the primary vector space model (VSM)
        emb1_dim: An int, the dimensionality of the vectors in the primary VSM
        vocab_size2: An int, the number of words in the secondary VSM, if one is provided
        emb2_dim: An int, the dimensionality of the vectors in the secondary VSM
        max_seq_length: The maximum length of the data sequences (used in LSTM to save computational resources)
        n_hidden: An int, the size of the individual layers in the LSTMs
        dropout: A float, the probability of dropping the activity of a neuron (dropout)
    Return:
        model: tf.keras.Model()
    """
    inputs1 = keras.Input(shape=(max_seq_length,), name='word_ids1')
    # inputs2 = keras.Input(shape=(max_seq_length,), name='word_ids2')
    # indices = keras.Input(shape=(None,), name="gold_indices", dtype="int32")
    mask = keras.Input(shape=(None,), name="gold_indices", dtype="int32")
    # possible_senses = keras.Input(shape=(None,), name="possible_senses", dtype="int32")
    emb_layer = keras.layers.Embedding(vocab_size1, emb1_dim,
                                       embeddings_initializer=keras.initializers.Constant(embeddings1),
                                       trainable=False)
    emb_inputs = emb_layer(inputs1)
    # emb_possibles = emb_layer(possible_senses)
    # if vocab_size2 > 0:
    #     emb_layer2 = keras.layers.Embedding(vocab_size2, emb2_dim,
    #                                         embeddings_initializer=keras.initializers.Constant(embeddings2),
    #                                         trainable=False)(inputs2)
    #     emb_layer = keras.layers.concatenate(emb_layer, emb_layer2)
    bilstm = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden))(emb_inputs)
    dropout = keras.layers.Dropout(dropout)(bilstm)
    outputs = keras.layers.Dense(output_dim, activation='relu')(dropout)
    outputs = tf.boolean_mask(outputs, mask)
    # outputs = tf.gather(outputs, indices)
    if method == "classification":
        outputs = keras.layers.Dense(output_dim, activation='softmax')(outputs)
        # outputs = keras.layers.Softmax(outputs)
    # elif method == "context_embedding":
    #     outputs = MetricLoggingLayer()([outputs, )
    model = keras.Model(inputs1, mask, outputs)
    return model




