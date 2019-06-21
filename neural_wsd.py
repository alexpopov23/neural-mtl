import argparse
import pickle

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.adam import Adam

import model
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
    lemma2synsets, max_synsets = read_data.get_wordnet_lexicon(args.lexicon_path)

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
                                                                                           wsd_method=args.wsd_method)
        train_input_ids, train_indices, train_gold_ids, train_len = read_data.get_ids(train_data,
                                                                                      emb1_src2id,
                                                                                      synset2id,
                                                                                      args.embeddings1_input)
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
                                                        wsd_method=args.wsd_method)
        test_input_ids, test_indices, test_gold_ids, test_len  = read_data.get_ids(test_data,
                                                                                   emb1_src2id,
                                                                                   synset2id,
                                                                                   args.embeddings1_input)
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
                                                       wsd_method=args.wsd_method)
        dev_input_ids, dev_indices, dev_gold_ids, dev_len = read_data.get_ids(dev_data,
                                                                              emb1_src2id,
                                                                              synset2id,
                                                                              args.embeddings1_input)

    dataset = tf.data.Dataset.range(train_len)
    dataset = dataset.map(lambda x: read_data.get_sequence(x, train_input_ids, train_indices, train_gold_ids, embeddings1))
    dataset = dataset.shuffle(10000)
    dataset = dataset.padded_batch(int(args.batch_size),
                                   padded_shapes=( (    ( (int(args.max_seq_length)), (int(args.max_seq_length)) ),
                                                        (int(args.max_seq_length), int(args.embeddings1_dim)) ) ),
                                   padding_values=( (0, False), 0.0) )

    # Create the model architecture
    if args.wsd_method == "classification":
        output_dim = len(synset2id)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
    elif args.wsd_method == "context_embedding":
        output_dim = int(args.embeddings1_dim)
        loss = "mse"
        metrics = [keras.losses.CosineSimilarity()] # which axis should be used ???
    model = model.get_model(args.wsd_method,
                            embeddings1,
                            output_dim,
                            int(args.max_seq_length),
                            int(args.n_hidden),
                            float(args.dropout))
    model.summary()
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss=loss,
    #               metrics=metrics,
    #               sample_weight_mode="temporal")
    # model.fit(dataset, epochs=2)

    # optimizer = keras.optimizers.Adam()
    optimizer = Adam()
    loss_fn = keras.losses.MeanSquaredError()
    metric_fn = keras.metrics.CosineSimilarity()
    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))
        step = 0
        for (x_batch_train, mask), y_batch_train in dataset.__iter__():
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train)
                outputs = tf.boolean_mask(outputs, mask)
                true_preds = tf.boolean_mask(y_batch_train, mask)
                loss_value = loss_fn(true_preds, outputs)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            metric_value = metric_fn(true_preds, outputs)
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * int(args.batch_size)))
            step += 1
        train_acc = metric_fn.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        metric_fn.reset_states()

        # # Run a validation loop at the end of each epoch.
        # for x_batch_val, y_batch_val in val_dataset:
        #     val_logits = model(x_batch_val)
        #     # Update val metrics
        #     val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        # print('Validation acc: %s' % (float(val_acc),))
