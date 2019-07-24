import argparse
import pickle
import sys

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.adam import Adam

import models
from data_ops import read_data, load_embeddings

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or evaluate a neural WSD model.', fromfile_prefix_chars='@')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-dev_data_path', dest='dev_data_path', required=False,
                        help='The path to the gold corpus used for development.')
    parser.add_argument("-dev_data_format", dest="dev_data_format", required=False, default="uef",
                        help="Specifies the format of the development corpus. Options: naf, uef")
    parser.add_argument('-dropout', dest='dropout', required=False, default="0",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-embeddings1_path', dest='embeddings1_path', required=True,
                        help='The path to the pretrained model with the primary embeddings.')
    parser.add_argument('-embeddings2_path', dest='embeddings2_path', required=False,
                        help='The path to the pretrained model with the additional embeddings.')
    parser.add_argument('-embeddings1_case', dest='embeddings1_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings2_case', dest='embeddings2_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings1_dim', dest='embeddings1_dim', required=False, default=300,
                        help='Size of the primary embeddings.')
    parser.add_argument('-embeddings2_dim', dest='embeddings2_dim', required=False, default=0,
                        help='Size of the additional embeddings.')
    parser.add_argument('-embeddings1_input', dest='embeddings1_input', required=False, default="wordform",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-embeddings2_input', dest='embeddings2_input', required=False, default="lemma",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.2,
                        help='How fast the network should learn.')
    parser.add_argument('-lexicon_path', dest='lexicon_path', required=False,
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-max_seq_length', dest='max_seq_length', required=False, default=63,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-mode', dest='mode', required=False, default="train",
                        help="Is this is a training, evaluation or application run? Options: train, evaluation, "
                             "application")
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-pos_filter', dest='pos_filter', required=False, default="False",
                        help='Whether to use POS information to filter out irrelevant synsets.')
    parser.add_argument('-pos_tagset', dest='pos_tagset', required=False, default="coarsegrained",
                        help='Whether the POS tags should be converted. Options are: coarsegrained, finegrained.')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved, or path to the folder with a saved model.')
    parser.add_argument('-sensekey2synset_path', dest='sensekey2synset_path', required=False,
                        help='Path to mapping between sense annotations in the corpus and synset IDs in WordNet.')
    parser.add_argument('-test_data_path', dest='test_data_path', required=False,
                        help='The path to the gold corpus used for testing.')
    parser.add_argument("-test_data_format", dest="test_data_format", required=False, default="uef",
                        help="Specifies the format of the evaluation corpus. Options: naf, uef")
    parser.add_argument('-train_data_path', dest='train_data_path', required=False,
                        help='The path to the gold corpus used for training.')
    parser.add_argument("-train_data_format", dest="train_data_format", required=False, default="uef",
                        help="Specifies the format of the training corpus. Options: naf, uef")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default=100001,
                        help='How many iterations the network should train for.')
    parser.add_argument('-wsd_method', dest='wsd_method', required=True,
                        help='Which method for WSD? Options: classification, context_embedding, multitask')

    # Load the embeddings model(s)
    args = parser.parse_args()
    embeddings1, emb1_src2id, emb1_id2src = load_embeddings.load_gensim(args.embeddings1_path)
    embeddings1 = tf.convert_to_tensor(embeddings1)
    if args.embeddings2_path is not None:
        embeddings2, emb2_src2id, emb2_id2src = load_embeddings.load_gensim(args.embeddings2_path)
    else:
        embeddings2, emb2_src2id, emb2_id2src = [], None, None
    if args.train_data_format == "uef" or args.test_data_format == "uef":
        sensekey2synset = pickle.load(open(args.sensekey2synset_path, "rb"))
    lemma2synsets, lemma2synset_ids, max_synsets = read_data.get_wordnet_lexicon(args.lexicon_path,
                                                                                 emb1_src2id,
                                                                                 pos_filter=args.pos_filter)
    # lemma2synset_ids = {lemma: [emb1_src2id[synset] for synset in synsets] if synset in emb1_src2id else emb1_src2id["<UNK>"]
    #                     for lemma, synsets in lemma2synsets.items() }

    # Read the data sets
    if args.train_data_format == "naf":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_naf(args.train_data_path,
                                                                                           lemma2synsets,
                                                                                           for_training=True,
                                                                                           wsd_method=args.wsd_method,
                                                                                           pos_tagset=args.pos_tagset)
    elif args.train_data_format == "uef":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_uef(args.train_data_path,
                                                                                           sensekey2synset,
                                                                                           lemma2synsets,
                                                                                           for_training=True,
                                                                                           wsd_method=args.wsd_method,
                                                                                           pos_filter=args.pos_filter)
        train_input_ids, train_input_lemmas, train_indices, train_gold_ids, train_gold_idxs, train_len = read_data.get_ids(
            train_data,
            emb1_src2id,
            synset2id,
            lemma2synsets,
            input_format=args.embeddings1_input,
            method=args.wsd_method,
            pos_filter=args.pos_filter)
    synID2embID = read_data.get_lookup_table(synset2id, emb1_src2id) # maps the synset IDs to the the embeddings IDs
    if args.test_data_format == "naf":
        test_data, _, _, _, _ = read_data.read_data_naf(args.test_data_path,
                                                        lemma2synsets,
                                                        lemma2id=lemma2id,
                                                        known_lemmas=known_lemmas,
                                                        synset2id=synset2id,
                                                        for_training=False,
                                                        wsd_method=args.wsd_method,
                                                        pos_tagset=args.pos_tagset)
    elif args.test_data_format == "uef":
        test_data, _, _, _, _ = read_data.read_data_uef(args.test_data_path,
                                                        sensekey2synset,
                                                        lemma2synsets,
                                                        lemma2id=lemma2id,
                                                        known_lemmas=known_lemmas,
                                                        synset2id=synset2id,
                                                        for_training=False,
                                                        wsd_method=args.wsd_method,
                                                        pos_filter=args.pos_filter)
        test_input_ids, test_input_lemmas, test_indices, test_gold_ids, test_gold_idxs, test_len  = \
            read_data.get_ids(test_data,
                              emb1_src2id,
                              synset2id,
                              lemma2synsets,
                              input_format=args.embeddings1_input,
                              method=args.wsd_method,
                              pos_filter=args.pos_filter)
    if args.dev_data_format == "naf":
        dev_data, _, _, _, _ = read_data.read_data_naf(args.dev_data_path,
                                                       lemma2synsets,
                                                       lemma2id=lemma2id,
                                                       known_lemmas=known_lemmas,
                                                       synset2id=synset2id,
                                                       for_training=False,
                                                       wsd_method=args.wsd_method,
                                                       pos_tagset=args.pos_tagset)
    elif args.dev_data_format == "uef":
        dev_data, _, _, _, _ = read_data.read_data_uef(args.dev_data_path,
                                                       sensekey2synset,
                                                       lemma2synsets,
                                                       lemma2id=lemma2id,
                                                       known_lemmas=known_lemmas,
                                                       synset2id=synset2id,
                                                       for_training=False,
                                                       wsd_method=args.wsd_method,
                                                       pos_filter=args.pos_filter)
        dev_input_ids, dev_input_lemmas, dev_indices, dev_gold_ids, dev_gold_idxs, dev_len = \
            read_data.get_ids(dev_data,
                              emb1_src2id,
                              synset2id,
                              lemma2synsets,
                              input_format=args.embeddings1_input,
                              method=args.wsd_method,
                              pos_filter=args.pos_filter)

    # Create the Dataset objects and configure them
    if args.wsd_method == "classification":
        output_dim = len(synset2id)
    elif args.wsd_method == "context_embedding":
        output_dim = int(args.embeddings1_dim)
    elif args.wsd_method == "multitask":
        output_dim = (len(synset2id), int(args.embeddings1_dim))
    max_seq_length = int(args.max_seq_length)
    if args.wsd_method == "multitask":
        padded_shapes = ((((max_seq_length),
                           (max_seq_length),
                           (max_seq_length),
                           (max_seq_length)),
                          ((max_seq_length, output_dim[0]), (max_seq_length, output_dim[1]))))
        padding_values = ((0, "<PAD>", 0., -1), (0.0, 0.0))
    else:
        padded_shapes = ( ( ( (max_seq_length),
                              (max_seq_length),
                              (max_seq_length),
                              (max_seq_length) ),
                            (max_seq_length, output_dim) ) )
        padding_values = ((0, "<PAD>", 0., -1), 0.0)
    train_dataset = tf.data.Dataset.range(train_len)
    train_dataset = train_dataset.map(lambda x: read_data.get_sequence(x,
                                                                       train_input_ids,
                                                                       train_input_lemmas,
                                                                       train_indices,
                                                                       train_gold_ids,
                                                                       train_gold_idxs,
                                                                       embeddings1,
                                                                       output_dim,
                                                                       args.wsd_method,
                                                                       synID2embID))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.padded_batch(int(args.batch_size),
                                               padded_shapes=padded_shapes,
                                               padding_values=padding_values)
    dev_dataset = tf.data.Dataset.range(dev_len)
    dev_dataset = dev_dataset.map(lambda x: read_data.get_sequence(x,
                                                                   dev_input_ids,
                                                                   dev_input_lemmas,
                                                                   dev_indices,
                                                                   dev_gold_ids,
                                                                   dev_gold_idxs,
                                                                   embeddings1,
                                                                   output_dim,
                                                                   args.wsd_method,
                                                                   synID2embID))
    dev_dataset = dev_dataset.shuffle(10000)
    dev_dataset = dev_dataset.padded_batch(int(args.batch_size),
                                               padded_shapes=padded_shapes,
                                               padding_values=padding_values)

    # Create the model architecture
    all_indices = tf.reshape(tf.range(int(args.batch_size)*max_seq_length, dtype="int32"),
                             [int(args.batch_size), max_seq_length])
    model = models.get_model(args.wsd_method,
                             embeddings1,
                             output_dim,
                             int(args.max_seq_length),
                             int(args.n_hidden),
                             float(args.dropout))
    model.summary()
    optimizer = Adam()
    if args.wsd_method == "classification":
        loss_fn = keras.losses.CategoricalCrossentropy()
        train_metric, val_metric = None, None
    elif args.wsd_method == "context_embedding":
        loss_fn = keras.losses.MeanSquaredError()
        train_metric, val_metric = keras.metrics.CosineSimilarity(), keras.metrics.CosineSimilarity()
    elif args.wsd_method == "multitask":
        loss_fn = models.MultitaskLoss(keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError())
        train_metric, val_metric = keras.metrics.CosineSimilarity(), keras.metrics.CosineSimilarity()
    train_accuracy, val_accuracy = tf.metrics.Accuracy(), tf.metrics.Accuracy()
    if args.wsd_method == "multitask":
        train_accuracy2, val_accuracy2 = tf.metrics.Accuracy(), tf.metrics.Accuracy()
    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))
        step = 0
        for (x_batch_train, x_batch_lemmas, mask, true_idxs), y_batch_train in train_dataset.__iter__():
            with tf.GradientTape() as tape:
                lemmas = tf.boolean_mask(x_batch_lemmas, mask)
                if args.wsd_method == "classification":
                    possible_synsets = [[synset2id[syn] for syn in lemma2synsets[lemma.numpy().decode("utf-8")]]
                                        for lemma in lemmas]
                elif args.wsd_method == "context_embedding":
                    possible_synsets = [lemma2synset_ids[lemma.numpy()] for lemma in lemmas]
                elif args.wsd_method == "multitask":
                    possible_synsets = ([[synset2id[syn] for syn in lemma2synsets[lemma.numpy().decode("utf-8")]]
                                         for lemma in lemmas],
                                        [lemma2synset_ids[lemma.numpy()] for lemma in lemmas])
                outputs = model((x_batch_train, mask))
                if args.wsd_method == "classification" or args.wsd_method == "context_embedding":
                    outputs = tf.boolean_mask(outputs, mask)
                    true_preds = tf.boolean_mask(y_batch_train, mask)
                elif args.wsd_method == "multitask":
                    outputs = (tf.boolean_mask(outputs[0], mask), tf.boolean_mask(outputs[1], mask))
                    true_preds = (tf.boolean_mask(y_batch_train[0], mask), tf.boolean_mask(y_batch_train[1], mask))
                loss_value = loss_fn(true_preds, outputs)
                if train_metric is not None:
                    if args.wsd_method == "context_embedding":
                        train_metric.update_state(true_preds, outputs)
                    elif args.wsd_method == "multitask":
                        train_metric.update_state(true_preds[1], outputs[1])
                if step % 50 == 0:
                    true_idxs = tf.boolean_mask(true_idxs, mask)
                    accuracy = models.accuracy(outputs, possible_synsets, embeddings1, true_idxs, train_accuracy,
                                               args.wsd_method, train_accuracy2)
                    train_accuracy.reset_states()
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 50 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Accuracy on last batch is: %s' % (accuracy[0]))
                if args.wsd_method == "multitask":
                    print('Alternative accuracy (cosine similarity) on last batch is: %s' % (accuracy[1]))
                print('Seen so far: %s samples' % ((step + 1) * int(args.batch_size)))
            step += 1
        if args.wsd_method == "context_embdding" or args.wsd_method == "multitask":
            cosine_sim = train_metric.result()
            print('Cosine similarity for the iteration is: %s' % (float(cosine_sim)))
            train_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for (x_batch_dev, x_batch_lemmas, mask, true_idxs), y_batch_dev in dev_dataset.__iter__():
            lemmas = tf.boolean_mask(x_batch_lemmas, mask)
            if args.wsd_method == "classification":
                possible_synsets = [[synset2id[syn] if syn in synset2id else synset2id["<UNK>"]
                                     for syn in lemma2synsets[lemma.numpy().decode("utf-8")]]
                                    for lemma in lemmas]
            elif args.wsd_method == "context_embedding":
                possible_synsets = [lemma2synset_ids[lemma.numpy()] for lemma in lemmas]
            elif args.wsd_method == "multitask":
                possible_synsets = ([[synset2id[syn] if syn in synset2id else synset2id["<UNK>"]
                                      for syn in lemma2synsets[lemma.numpy().decode("utf-8")]]
                                     for lemma in lemmas],
                                    [lemma2synset_ids[lemma.numpy()] for lemma in lemmas])
            outputs = model((x_batch_dev, mask))
            if args.wsd_method == "classification" or args.wsd_method == "context_embedding":
                outputs = tf.boolean_mask(outputs, mask)
                true_preds = tf.boolean_mask(y_batch_dev, mask)
            elif args.wsd_method == "multitask":
                outputs = (tf.boolean_mask(outputs[0], mask), tf.boolean_mask(outputs[1], mask))
                true_preds = (tf.boolean_mask(y_batch_dev[0], mask), tf.boolean_mask(y_batch_dev[1], mask))
            if val_metric is not None:
                if args.wsd_method == "context_embedding":
                    val_metric.update_state(true_preds, outputs)
                elif args.wsd_method == "multitask":
                    val_metric.update_state(true_preds[1], outputs[1])
            true_idxs = tf.boolean_mask(true_idxs, mask)
            accuracy = models.accuracy(outputs, possible_synsets, embeddings1, true_idxs, val_accuracy, args.wsd_method,
                                       val_accuracy2)
        val_cosine_sim = val_metric.result()
        val_metric.reset_states()
        print('Validation cosine similarity metric: %s' % (float(val_cosine_sim),))
        val_acc = val_accuracy.result()
        val_accuracy.reset_states()
        print('Validation accuracy: %s' % (float(val_acc),))
        if args.wsd_method == "multitask":
            val_acc2 = val_accuracy2.result()
            val_accuracy2.reset_states()
            print('Alternative validation accuracy (cosine similarity): %s' % (float(val_acc2)))


