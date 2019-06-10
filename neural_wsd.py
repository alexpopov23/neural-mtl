import argparse
import pickle

import tensorflow as tf
from tensorflow.python import keras

from data_ops import format_data, read_data, load_embeddings, evaluation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or evaluate a neural WSD model.', fromfile_prefix_chars='@')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-dev_data_path', dest='dev_data', required=False,
                        help='The path to the gold corpus used for development.')
    parser.add_argument("-dev_data_format", dest="dev_data_format", required=False, default="uef",
                        help="Specifies the format of the development corpus. Options: naf, uef")
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
    parser.add_argument('-keep_prob', dest='keep_prob', required=False, default="1",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.2,
                        help='How fast the network should learn.')
    parser.add_argument('-lexicon_path', dest='lexicon_path', required=False,
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-max_seq_length', dest='max_seq_length', required=False, default=63,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training, evaluation or application run? Options: train, evaluation, "
                             "application")
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-pos_classifier', dest='pos_classifier', required=False, default="False",
                        help='Should the system also perform POS tagging? Available only with classification.')
    parser.add_argument('-pos_tagset', dest='pos_tagset', required=False, default="coarsegrained",
                        help='Whether the POS tags should be converted. Options are: coarsegrained, finegrained.')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved, or path to the folder with a saved model.')
    parser.add_argument('-sensekey2synset_path', dest='sensekey2synset_path', required=False,
                        help='Path to mapping between sense annotations in the corpus and synset IDs in WordNet.')
    parser.add_argument('-test_data_path', dest='test_data', required=False,
                        help='The path to the gold corpus used for testing.')
    parser.add_argument("-test_data_format", dest="test_data_format", required=False, default="uef",
                        help="Specifies the format of the evaluation corpus. Options: naf, uef")
    parser.add_argument('-train_data_path', dest='train_data', required=False,
                        help='The path to the gold corpus used for training.')
    parser.add_argument("-train_data_format", dest="train_data_format", required=False, default="uef",
                        help="Specifies the format of the training corpus. Options: naf, uef")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default=100001,
                        help='How many iterations the network should train for.')
    parser.add_argument('-wsd_classifier', dest='wsd_classifier', required=True,
                        help='Should the system perform WSD?')
    parser.add_argument('-wsd_method', dest='wsd_method', required=True,
                        help='Which method for WSD? Options: classification, context_embedding, multitask')

    args = parser.parse_args()
    embeddings1, emb1_src2id, emb1_id2src = load_embeddings.load_gensim(args.embeddings1_path)
    if args.embeddings2_path is not None:
        embeddings2, emb2_src2id, emb2_id2src = load_embeddings.load_gensim(args.embeddings2_path)
    else:
        embeddings2, emb2_src2id, emb2_id2src = [], None, None
    if args.train_data_format == "uef" or args.test_data_format == "uef":
        sensekey2synset = pickle.load(open(args.sensekey2synset_path, "rb"))
    lemma2synsets = read_data.get_wordnet_lexicon(args.lexicon_path)

    # Read the data sets
    if args.train_data_format == "naf":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_naf(
            args.train_data_path, lemma2synsets, for_training=True, wsd_method=args.wsd_method, pos_tagset=pos_tagset)
    elif args.train_data_format == "uef":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_uef(
            args.train_data_path, sensekey2synset, lemma2synsets, for_training=True, wsd_method=args.wsd_method)
    if args.test_data_format == "naf":
        test_data, _, _, _, _ = read_data.read_data_naf(
            args.test_data_path, lemma2synsets, lemma2id=lemma2id, known_lemmas=known_lemmas,
            synset2id=synset2id, for_training=False, wsd_method=args.wsd_method, pos_tagset=args.pos_tagset)
    elif args.test_data_format == "uef":
        test_data, _, _, _, _ = read_data.read_data_uef(
            args.test_data_path, sensekey2synset, lemma2synsets, lemma2id=lemma2id,
            known_lemmas=known_lemmas, synset2id=synset2id, for_training=False, wsd_method=args.wsd_method)
    if args.dev_data_format == "naf":
        dev_data, _, _, _, _ = read_data.read_data_naf(
            args.dev_data_path, lemma2synsets, lemma2id=lemma2id, known_lemmas=known_lemmas,
            synset2id=synset2id, for_training=False, wsd_method=args.wsd_method, pos_tagset=args.pos_tagset)
    elif args.dev_data_format == "uef":
        dev_data, _, _, _, _ = read_data.read_data_uef(
            args.dev_data_path, sensekey2synset, lemma2synsets, lemma2id=lemma2id,
            known_lemmas=known_lemmas, synset2id=synset2id, for_training=False, wsd_method=args.wsd_method)

    (test_inputs1,
     test_inputs2,
     test_sequence_lengths,
     test_labels_classif,
     test_labels_context,
     test_labels_pos,
     test_indices,
     test_target_lemmas,
     test_synsets_gold,
     test_pos_filters) = format_data.format_data(
        test_data, emb1_src2id, args.embeddings1_input, args.embeddings1_case, synset2id, args.max_seq_length, embeddings1,
        emb2_src2id, args.embeddings2_input, args.embeddings2_case, args.embeddings1_dim, pos_types, args.pos_classifier, args.wsd_method)

    (dev_inputs1,
     dev_inputs2,
     dev_sequence_lengths,
     dev_labels_classif,
     dev_labels_context,
     dev_labels_pos,
     dev_indices,
     dev_target_lemmas,
     dev_synsets_gold,
     dev_pos_filters) = format_data.format_data(
        test_data, emb1_src2id, args.embeddings1_input, args.embeddings1_case, synset2id, args.max_seq_length, embeddings1,
        emb2_src2id, args.embeddings2_input, args.embeddings2_case, args.embeddings1_dim, pos_types, args.pos_classifier, args.wsd_method)



    '''
    model = None
    if wsd_method == "classification":
        output_dimension = len(synset2id)
        model = architectures.classifier.ClassifierSoftmax(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_classif, wsd_classifier, pos_classifier, len(pos_types),
            test_labels_pos)
    elif wsd_method == "context_embedding":
        output_dimension = embeddings1_dim
        model = architectures.context_embedding.ContextEmbedder(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_context, wsd_classifier, pos_classifier, len(pos_types),
            test_labels_pos)
    elif wsd_method == "multitask":
        output_dimension = len(synset2id)
        model = architectures.multitask_wsd.MultitaskWSD(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_classif, test_labels_context, wsd_classifier,
            pos_classifier, len(pos_types), test_labels_pos)

    session = tf.Session()
    saver = tf.train.Saver()
    if mode != "evaluation":
        model_path = os.path.join(save_path, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if wsd_method == "multitask":
            model_path_context = os.path.join(save_path, "model_context")
            if not os.path.exists(model_path_context):
                os.makedirs(model_path_context)
    if mode == "evaluation":
        with open(os.path.join(save_path, "checkpoint"), "r") as f:
            for line in f.readlines():
                if line.split()[0] == "model_checkpoint_path:":
                    model_checkpoint_path = line.split()[1].rstrip("\n")
                    model_checkpoint_path = model_checkpoint_path.strip("\"")
                    break
        saver.restore(session, model_checkpoint_path)
        # saver.restore(session, save_path)
        app_data = test_data
        match_wsd_classif_total, eval_wsd_classif_total, match_classif_wsd, eval_classif_wsd = [0, 0, 0, 0]
        match_wsd_context_total, eval_wsd_context_total, match_wsd_context, eval_wsd_context = [0, 0, 0, 0]
        match_pos_total, eval_pos_total, match_pos, eval_pos = [0, 0, 0, 0]
        for step in range(len(app_data) / batch_size + 1):
            offset = (step * batch_size) % (len(app_data))
            if offset + batch_size > len(app_data):
                buffer =  (offset + batch_size) - len(app_data)
                zero_element = ["UNK", "UNK", ".", ["unspecified"], [-1]]
                zero_sentence = batch_size * [zero_element]
                buffer_data = buffer * [zero_sentence]
                app_data.extend(buffer_data)
            (inputs1,
             inputs2,
             seq_lengths,
             labels_classif,
             labels_context,
             labels_pos,
             indices,
             target_lemmas,
             synsets_gold,
             pos_filters) = format_data.new_batch(
                offset, batch_size, app_data, emb1_src2id, embeddings1_input, embeddings1_case,
                synset2id, max_seq_length, embeddings1, emb2_src2id, embeddings2_input,
                embeddings2_case, embeddings1_dim, pos_types, pos_classifier, wsd_method)
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, pos_classifier, "evaluation", wsd_method)
            if wsd_method == "classification":
                _, _, [match_classif_wsd, eval_classif_wsd, match_pos, eval_pos] = evaluation.accuracy_classification(
                    fetches[0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, fetches[1], labels_pos)
            elif wsd_method == "context_embedding":
                _, [match_wsd_context, eval_wsd_context] = evaluation.accuracy_cosine_distance(
                    fetches[0], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
            elif wsd_method == "multitask":
                _, _, [match_classif_wsd, eval_classif_wsd, _, _] = evaluation.accuracy_classification(
                    fetches[0][0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, labels_pos)
                _, [match_wsd_context, eval_wsd_context] = evaluation.accuracy_cosine_distance(
                    fetches[0][1], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
            match_wsd_classif_total += match_classif_wsd
            eval_wsd_classif_total += eval_classif_wsd
            match_wsd_context_total += match_wsd_context
            eval_wsd_context_total += eval_wsd_context
            match_pos_total += match_pos
            eval_pos_total += eval_pos
        if wsd_method == "classification":
            print "Accuracy for WSD (CLASSIFICATION) is " + \
                  str((100.0 * match_wsd_classif_total) / eval_wsd_classif_total) + "%"
        elif wsd_method == "context_embedding":
            print "Accuracy for WSD (CONTEXT_EMBEDDING) is " + \
                  str((100.0 * match_wsd_context_total) / eval_wsd_context_total) + "%"
        elif wsd_method == "multitask":
            print "Accuracy for WSD (CLASSIFICATION) is " + \
                  str((100.0 * match_wsd_classif_total) / eval_wsd_classif_total) + "%"
            print "Accuracy for WSD (CONTEXT_EMBEDDING) is " + \
                  str((100.0 * match_wsd_context_total) / eval_wsd_context_total) + "%"
        if pos_classifier is True:
            print "Accuracy for POS tagging is " + \
                  str((100.0 * match_pos_total) / eval_pos_total) + "%"
        exit()
    else:
        init = tf.global_variables_initializer()
        feed_dict = {model.emb1_placeholder: embeddings1}
        if len(embeddings2) > 0:
            feed_dict.update({model.emb2_placeholder: embeddings2})
        session.run(init, feed_dict=feed_dict)
    print "Start of training"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    results = open(os.path.join(args.save_path, 'results.txt'), "a", 0)
    results.write(str(args) + '\n\n')
    batch_loss = 0
    best_accuracy_wsd, best_accuracy_context = 0.0, 0.0
    for step in range(training_iterations):
        offset = (step * batch_size) % (len(train_data) - batch_size)
        (inputs1,
         inputs2,
         seq_lengths,
         labels_classif,
         labels_context,
         labels_pos,
         indices,
         target_lemmas,
         synsets_gold,
         pos_filters) = format_data.new_batch(
            offset, batch_size, train_data, emb1_src2id, embeddings1_input, embeddings1_case,
            synset2id, max_seq_length, embeddings1, emb2_src2id, embeddings2_input,
            embeddings2_case, embeddings1_dim, pos_types, pos_classifier, wsd_method)
        test_accuracy_wsd, test_accuracy_context = 0.0, 0.0
        if step % 100 == 0:
            print "Step number " + str(step)
            results.write('EPOCH: %d' % step + '\n')
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, pos_classifier, "validation", wsd_method)
            if fetches[1] is not None:
                batch_loss += fetches[1]
            results.write('Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss / 100.0) + '\n')
            if wsd_method == "classification":
                minibatch_accuracy_wsd, minibatch_accuracy_pos, _ = evaluation.accuracy_classification(
                    fetches[2], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, fetches[4], labels_pos)
                test_accuracy_wsd, test_accuracy_pos, _ = evaluation.accuracy_classification(
                    fetches[5], test_target_lemmas, test_synsets_gold, test_pos_filters, synset2id, lemma2synsets,
                    known_lemmas, wsd_classifier, pos_classifier, fetches[6], test_labels_pos)
                if wsd_classifier is True:
                    results.write('Minibatch WSD accuracy: ' + str(minibatch_accuracy_wsd) + '\n')
                    results.write('Validation WSD accuracy: ' + str(test_accuracy_wsd) + '\n')
                if pos_classifier is True:
                    results.write('Minibatch POS tagging accuracy: ' + str(minibatch_accuracy_pos) + '\n')
                    results.write('Validation POS tagging accuracy: ' + str(test_accuracy_pos) + '\n')
            elif wsd_method == "context_embedding":
                minibatch_accuracy_wsd, _ = evaluation.accuracy_cosine_distance(
                    fetches[2], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
                test_accuracy_wsd, _ = evaluation.accuracy_cosine_distance(
                    fetches[5], test_target_lemmas, test_synsets_gold, test_pos_filters, lemma2synsets, embeddings1,
                    emb1_src2id)
                results.write('Minibatch WSD accuracy: ' + str(minibatch_accuracy_wsd) + '\n')
                results.write('Validation WSD accuracy: ' + str(test_accuracy_wsd) + '\n')
            elif wsd_method == "multitask":
                minibatch_accuracy_wsd, _, _ = evaluation.accuracy_classification(
                    fetches[2][0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, labels_pos)
                test_accuracy_wsd, _, _ = evaluation.accuracy_classification(
                    fetches[5][0], test_target_lemmas, test_synsets_gold, test_pos_filters, synset2id, lemma2synsets,
                    known_lemmas, wsd_classifier, pos_classifier, test_labels_pos)
                minibatch_accuracy_context, _ = evaluation.accuracy_cosine_distance(
                    fetches[2][1], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
                test_accuracy_context, _ = evaluation.accuracy_cosine_distance(
                    fetches[5][1], test_target_lemmas, test_synsets_gold, test_pos_filters, lemma2synsets, embeddings1,
                    emb1_src2id)
                results.write('Minibatch WSD accuracy (CLASSIFICATION): ' + str(minibatch_accuracy_wsd) + '\n')
                results.write('Validation WSD accuracy (CLASSIFICATION): ' + str(test_accuracy_wsd) + '\n')
                results.write('Minibatch WSD accuracy (CONTEXT_EMBEDDING): ' + str(minibatch_accuracy_context) + '\n')
                results.write('Validation WSD accuracy (CONTEXT_EMBEDDING): ' + str(test_accuracy_context) + '\n')
            print "Validation accuracy: " + str(test_accuracy_wsd)
            if wsd_method == "multitask":
                print "Validation accuracy (CONTEXT_EMBEDDING): " + str(test_accuracy_context)
            batch_loss = 0.0
        else:
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, pos_classifier, "train", wsd_method)
            if fetches[1] is not None:
                batch_loss += fetches[1]
        if test_accuracy_wsd > best_accuracy_wsd:
            best_accuracy_wsd = test_accuracy_wsd
        if wsd_method == "multitask" and test_accuracy_context > best_accuracy_context:
            best_accuracy_context = test_accuracy_context
        if args.save_path is not None:
            if step == 100 or step > 100 and test_accuracy_wsd == best_accuracy_wsd:
                for file in os.listdir(model_path):
                    os.remove(os.path.join(model_path, file))
                saver.save(session, os.path.join(args.save_path, "model/model.ckpt"), global_step=step)
            if wsd_method == "multitask" and \
                    (step == 10000 or step > 10000 and test_accuracy_context == best_accuracy_context):
                for file in os.listdir(model_path_context):
                    os.remove(os.path.join(model_path_context, file))
                saver.save(session, os.path.join(args.save_path, "model_context/model_context.ckpt"), global_step=step)
    results.write('\n\n\n' + 'Best result is: ' + str(best_accuracy_wsd))
    if wsd_method == "multitask":
        results.write('\n\n\n' + 'Best result (CONTEXT_EMBEDDING) is: ' + str(best_accuracy_context))
    results.close()

    '''
