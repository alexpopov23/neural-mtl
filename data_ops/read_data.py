import collections
import os

import _elementtree
import xml.etree.ElementTree as ET

import tensorflow as tf

import globals


def get_wordnet_lexicon(lexicon_path, syn2id=None, pos_filter="False"):
    """Reads the WordNet dictionary

    Args:
        lexicon_path: A string, the path to the dictionary

    Returns:
        lemma2synsets: A dictionary, maps lemmas to synset IDs

    """
    lemma2synsets, lemma2synset_ids = {}, {}
    lexicon = open(lexicon_path, "r")
    max_synsets = 0
    for line in lexicon.readlines():
        fields = line.split(" ")
        lemma, synsets = fields[0], fields[1:]
        if len(synsets) > max_synsets:
            max_synsets = len(synsets)
        for i, entry in enumerate(synsets):
            synset = entry[:10].strip()
            if pos_filter == "True":
                pos = synset[-1]
                lemma_pos = lemma + "-" + pos
            else:
                lemma_pos = lemma
            if lemma_pos not in lemma2synsets:
                lemma2synsets[lemma_pos] = [synset]
                lemma2synset_ids[lemma_pos.encode('utf-8')] = [syn2id[synset] if synset in syn2id else syn2id["<UNK>"]]
            else:
                lemma2synsets[lemma_pos].append(synset)
                lemma2synset_ids[lemma_pos.encode('utf-8')].append(syn2id[synset] if synset in syn2id else syn2id["<UNK>"])
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    lemma2synset_ids = collections.OrderedDict(sorted(lemma2synset_ids.items()))
    return lemma2synsets, lemma2synset_ids, max_synsets


def get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id, synset2id):
    """Constructs mappings between lemmas and integer IDs, synsets and integerIDs

    Args:
        wsd_method: A string ("classification"|"context_embedding"|"multitask")
        lemma2synsets: A dictionary, maps lemmas to synset IDs
        known_lemmas: A set of lemmas seen in the training data
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty)
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs (empty)

    Returns:
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs

    """
    index_l, index_s = 0, 0
    # if wsd_method == "classification" or wsd_method == "multitask":
    # synset2id['notseen-n'], synset2id['notseen-v'], synset2id['notseen-a'], synset2id['notseen-r'] = 0, 1, 2, 3
    synset2id["<UNK>"] = 0
    index_s = 1
    for lemma, synsets in lemma2synsets.items():
        if (wsd_method == "classification" or wsd_method == "multitask") and lemma not in known_lemmas:
            continue
        lemma2id[lemma] = index_l
        index_l += 1
        for synset in synsets:
            if synset not in synset2id:
                synset2id[synset] = index_s
                index_s += 1
    return lemma2id, synset2id


def add_synset_ids(wsd_method, data, known_lemmas, synset2id):
    """Adds integer IDs for the synset annotations of words in data

    Args:
        wsd_method: A string ("classification"|"context_embedding"|"multitask")
        data:   A list of lists; each sentence contains "words" represented
                in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs

    Returns:
        data:   A list of lists; each sentence contains "words" represented
                in the format: [wordform, lemma, POS, [synset1, ..., synsetN], [synsetID1, ..., synsetIDN]]

    """
    for sentence in data:
        for word in sentence:
            synsets = word[3]
            if synsets[0] != "unspecified":
                synset_ids = []
                lemma = word[1]
                pos = word[2]
                if (wsd_method == "classification" or wsd_method == "multitask") and lemma not in known_lemmas:
                    if pos == "NOUN" or pos in globals.pos_map and globals.pos_map[pos] == "NOUN":
                        synset_ids.append(synset2id['notseen-n'])
                    elif pos == "VERB" or pos in globals.pos_map and globals.pos_map[pos] == "VERB":
                        synset_ids.append(synset2id['notseen-v'])
                    elif pos == "ADJ" or pos in globals.pos_map and globals.pos_map[pos] == "ADJ":
                        synset_ids.append(synset2id['notseen-a'])
                    elif pos == "ADV" or pos in globals.pos_map and globals.pos_map[pos] == "ADV":
                        synset_ids.append(synset2id['notseen-r'])
                else:
                    for synset in synsets:
                        synset_ids.append(synset2id[synset])
                word.append(synset_ids)
            else:
                word.append([-1])
    return data


def read_naf_file(path, pos_tagset, pos_types):
    """Reads file in NAF format

    Args:
        path: A string, the path to the NAF file
        pos_tagset: A string, indicates whether POS tags should be coarse- or fine-grained
        pos_types: A dictionary, maps known POS tags to unique integer IDs

    Returns:
        sentences: A list of lists; each sentence contains "words" represented
                   in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]

    """
    tree = _elementtree.parse(path)
    doc = tree.getroot()
    text = doc.find("text")
    wfs = text.findall("wf")
    corpus = {}
    known_lemmas = set()
    pos_count = len(pos_types)
    for wf in wfs:
        wf_id = int(wf.get("id")[1:])
        wf_text = wf.text
        wf_sent = wf.get("sent")
        corpus[wf_id] = [wf_sent, wf_text]
    terms = doc.find("terms")
    for term in terms.findall("term"):
        lemma = term.get("lemma")
        pos = term.get("pos")
        if pos in globals.pos_normalize:
            pos = globals.pos_normalize[pos]
        if pos_tagset == "coarsegrained":
            if pos in globals.pos_map:
                pos = globals.pos_map[pos]
        if pos not in pos_types:
            pos_types[pos] = pos_count
            pos_count += 1
        id = int(term.find("span").find("target").get("id")[1:])
        synset = "unspecified"
        ext_refs = term.find("externalReferences")
        if ext_refs is not None:
            for extRef in ext_refs.findall("externalRef"):
                resource = extRef.get("resource")
                if resource == "WordNet-eng30" or resource == "WordNet-3.0":
                    reftype = extRef.get("reftype")
                    if reftype == "synset" or reftype == "ilidef":
                        synset = extRef.get("reference")[-10:]
                        if lemma not in known_lemmas:
                            known_lemmas.add(lemma)
        corpus[id].extend([lemma, pos, [synset]])
    corpus = collections.OrderedDict(sorted(corpus.items()))
    sentences = []
    current_sentence = []
    sent_counter = 1
    for word in corpus.iterkeys():
        if len(corpus[word]) == 2:
            lemma = corpus[word][1]
            corpus[word].extend([lemma, ".", ["unspecified"]])
        if int(corpus[word][0]) == sent_counter:
            current_sentence.append(corpus[word][1:])
        else:
            if sent_counter != 0:
                sentences.append(current_sentence)
            sent_counter += 1
            current_sentence = []
            current_sentence.append(corpus[word][1:])
    sentences.append(current_sentence)
    return sentences, known_lemmas


def read_data_naf(path, lemma2synsets, lemma2id={}, known_lemmas=set(), synset2id={}, for_training=True,
                  wsd_method="classification", pos_tagset="coarsegrained"):
    """Reads folders with files in NAF format

    Args:
        path: A string, the path to the data folder
        lemma2synsets: A dictionary, mapping lemmas to lists of synset IDs
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty when reading training data)
        known_lemmas: A set of lemmas seen in the training data (empty when reading training data)
        synset2id: A dictionary, mapping synsets to integer IDs (empty when reading training data)
        for_training: A boolean, indicates whether the data is for training or testing
        wsd_method: A string, indicates the disamguation method used ("classification", "context_embedding", "multitask")
        pos_tagset: A string, indicates whether POS tags should be coarse- or fine-grained

    Returns:
        data: A list of lists; each sentence contains "words" represented
              in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set, all lemmas seen in training
        pos_types: A dictionary, all POS tags seen in training and their mappings to integer IDs
        synset2id: A dictionary, mapping synsets to integer IDs

    """
    data = []
    pos_types = {}
    for f in os.listdir(path):
        new_data, new_lemmas = read_naf_file(os.path.join(path, f), pos_tagset, pos_types)
        known_lemmas.update(new_lemmas)
        data.extend(new_data)
    pos_types["."] = len(pos_types)
    if for_training is True:
        lemma2id, synset2id = get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id,
                                                                    synset2id)
    data = add_synset_ids(wsd_method, data, known_lemmas, synset2id)
    return data, lemma2id, known_lemmas, pos_types, synset2id


def read_data_uef(path, sensekey2synset, lemma2synsets, lemma2id={}, known_lemmas=set(), synset2id={},
                  for_training=True, wsd_method="classification", pos_filter="False"):
    """Reads a corpus in the Universal Evaluation Framework (UEF) format

    Args:
        path: A string, the path to the data folder
        sensekey2synset: A dictionary, mapping sense IDs to synset IDs
        lemma2synsets: A dictionary, mapping lemmas to lists of synset IDs
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty when reading training data)
        known_lemmas: A set of lemmas seen in the training data (empty when reading training data)
        synset2id: A dictionary, mapping synsets to integer IDs (empty when reading training data)
        for_training: A boolean, indicates whether the data is for training or testing
        wsd_method: A string, indicates the disamguation method used ("classification", "context_embedding", "multitask")

    Returns:
        data: A list of lists; each sentence contains "words" represented
              in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set, all lemmas seen in training
        pos_types: A dictionary, all POS tags seen in training and their mappings to integer IDs
        synset2id: A dictionary, mapping synsets to integer IDs
    """
    data = []
    pos_types, pos_count = {}, 0
    path_data = ""
    path_keys = ""
    for f in os.listdir(path):
        if f.endswith(".xml"):
            path_data = f
        elif f.endswith(".txt"):
            path_keys = f
    codes2keys = {}
    f_codes2keys = open(os.path.join(path, path_keys), "r")
    for line in f_codes2keys.readlines():
        fields = line.strip().split()
        code = fields[0]
        keys = fields[1:]
        codes2keys[code] = keys
    tree = ET.parse(os.path.join(path, path_data))
    doc = tree.getroot()
    corpora = doc.findall("corpus")
    for corpus in corpora:
        texts = corpus.findall("text")
        count_double_synsets = 0
        for text in texts:
            sentences = text.findall("sentence")
            for sentence in sentences:
                current_sentence = []
                elements = sentence.findall(".//")
                for element in elements:
                    wordform = element.text
                    lemma = element.get("lemma")
                    pos = element.get("pos")
                    if for_training is True:
                        if pos_filter == "True":
                            pos_simple = globals.pos_map_simple[pos] if pos in globals.pos_map_simple else "func"
                            known_lemmas.add(lemma + "-" + pos_simple)
                        else:
                            known_lemmas.add(lemma)
                    if pos not in pos_types:
                        pos_types[pos] = pos_count
                        pos_count += 1
                    if element.tag == "instance":
                        synsets = [sensekey2synset[key] for key in codes2keys[element.get("id")]]
                    else:
                        synsets = ["<NONE>"]
                    if len(synsets) > 1:
                        count_double_synsets += 1
                    current_sentence.append([wordform, lemma, pos, synsets])
                data.append(current_sentence)
    if for_training is True:
        lemma2id, synset2id = get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id, synset2id)
    # data = add_synset_ids(wsd_method, data, known_lemmas, synset2id)
    print("The number of double gold labels is %d" % (int(count_double_synsets),))
    return data, lemma2id, known_lemmas, pos_types, synset2id

def get_ids(data, input2id, synset2id, lemma2synsets, input_format="lemma", method="context_embedding", max_length=100,
            pos_filter="False"):
    """Converts the string data to numerical IDs per word, to be passed to the model during training/evaluation.

    Args:
        data: A list; sentence lists of words, where each one is represented as [wordform, lemma, POS, [synsets]]
        input2id: A dictionary; maps input string to embedding ID
        synset2id: A dictionary; maps synsets to integer IDs in the embeddings
        lemma2synsets: A dictionary; retrieves the synsets associated with a lemma
        input_format: A string; denotes the kind of input to be read off the data (lemma or wordform)
        max_length: An int; the maximum length of the sentences to be analyzed

    Returns:
        input_ids: A RaggedTensor of ints; the IDs for the input strings
        input_lemmas: A RaggedTensor of strings; the lemmas of the disambiguated words per sentence
        mask: A boolean RaggedTensor; masks the inputs that are not to be disambiguated
        gold_ids: A RaggedTensor with the gold synset IDs (words no to be disambiguated are marked with 0)
        gold_idxs: A RaggedTensor of ints; the ints are the positions of the synsets in the lexicon (1st,..,Nth)
        data_len: An int; the length of the dataset
    """
    input_ids, input_lemmas, mask, gold_ids, gold_idxs = [], [], [], [], []
    for sentence in data:
        # IMPORTANT: With max_length = 100 about 50 sentences from SemCor are excluded from the dataset!
        if len(sentence) > max_length:
            continue
        current_sent, current_lemmas, current_mask, current_gold_ids, current_gold_idxs = [], [], [], [], []
        for i, word in enumerate(sentence):
            if input_format == "wordform":
                current_sent.append(input2id[word[0]] if word[0] in input2id else input2id["<UNK>"])
            elif input_format == "lemma":
                current_sent.append(input2id[word[1]] if word[1] in input2id else input2id["<UNK>"])
            if word[3][0] != "<NONE>":
                current_mask.append(True)
                if method == "classification":
                    # TODO Picking only the first synset in cases of multiple gold labels; need to think more on handling it.
                    # current_gold_ids.append([synset2id[synset] if synset in synset2id else synset2id["<UNK>"] for synset in word[3]])
                    current_gold_ids.append(synset2id[word[3][0]] if word[3][0] in synset2id else synset2id["<UNK>"])
                elif method == "context_embedding":
                    # current_gold_ids.append([input2id[synset] if synset in input2id else input2id["<UNK>"] for synset in word[3]])
                    current_gold_ids.append(input2id[word[3][0]] if word[3][0] in input2id else input2id["<UNK>"])
                # taking only the first synset from the gold labels (not a problem for Sem/Senseval, but inaccurate for SemCor
                if pos_filter == "True":
                    lemma_pos = word[1] + "-" + globals.pos_map_simple[word[2]]
                else:
                    lemma_pos = word[1]
                current_gold_idxs.append(lemma2synsets[lemma_pos].index(word[3][0]))
            else:
                current_mask.append(False)
                # current_gold_ids.append([0]) # '<UNK>' in both cases
                current_gold_ids.append(0)
                current_gold_idxs.append(-1)
            if pos_filter == "True":
                pos = globals.pos_map_simple[word[2]] if word[2] in globals.pos_map_simple else "func"
                current_lemmas.append(word[1] + "-" + pos)
            else:
                current_lemmas.append(word[1])
        input_ids.append(current_sent)
        mask.append(current_mask)
        gold_ids.append(current_gold_ids)
        gold_idxs.append(current_gold_idxs)
        input_lemmas.append(current_lemmas)
        data_len = len(input_ids)
    input_ids = tf.ragged.constant(input_ids, dtype="int32")
    input_lemmas = tf.ragged.constant(input_lemmas, dtype="string")
    indices = tf.ragged.constant(mask, dtype="bool")
    gold_ids = tf.ragged.constant(gold_ids, dtype="int32")
    gold_idxs = tf.ragged.constant(gold_idxs, dtype="int32")
    return input_ids, input_lemmas, indices, gold_ids, gold_idxs, data_len

def get_sequence(x, data, lemmas, mask, gold_labels, gold_idxs, embeddings, output_size, method):
    """Retrieves one training/evaluation batch, to be passed to the model.

    Args:
        x: A tensor scalar; index into the slice of data to be retrieved in the construction of a particular data sample
        data: A RaggedTensor; the input IDs
        lemmas: A RaggedTensor; the input lemmas (per word present in the lexicon)
        mask: A RaggedTensor; a boolean mask to filter out the words not to be disambiguated
        gold_labels: A RaggedTensor; the gold synset IDs
        gold_idxs: A RaggedTensor; the positions of the gold synsets in the lemma-synsets pairings in the lexicon
        embeddings: A Tensor; the embeddings used by the model

    Returns:
        ((sequence, lemmas, mask, idxs), labels): A nested structure of tensors; to be consumed by the Dataset object
    """
    sequence = tf.gather(data, x)
    lemmas = tf.gather(lemmas, x)
    mask = tf.gather(mask, x)
    labels = tf.gather(gold_labels, x)
    if method == "classification":
        labels = tf.one_hot(labels, output_size)
    elif method == "context_embedding":
        labels = tf.gather(embeddings, labels)
        # labels = tf.reduce_mean(labels, 1) # not necessary when using only 1 gold label
    idxs = tf.gather(gold_idxs, x)
    return ((sequence, lemmas, mask, idxs), labels)


