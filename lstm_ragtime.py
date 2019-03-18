#! /usr/bin/env python
""" This module prepares midi file data and feeds it to the neural
    network for training """

from sys import argv
import glob
import music21 as m21
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Reshape, LSTM, Dense, Input, Lambda, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt

# directory where is the midi file corpus
MIDI_CORPUS_DIRECTORY = "joplin"
# file name to save precomputed notes (notes extracted from midi file corpus)
NOTES_FILE = "data/notes"
# file name to save precomputed sequences (to train the network)
VOCAB_FILE = "data/vocab"
# file name to save layers of the model
LAYERS_FILE = "data/layers"
# file name format to save pretrained network weights
WEIGHT_FILE_FMT = "weight/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# file name to load pretrained network weights
WEIGHT_FILE = "weight/weights-improvement-200-18.9266.hdf5"

# offset variation from one note to the next (1=quarter, 0.25=16th note)
OFFSET_STEP = 0.25
# length of a music sequence
SEQ_LENGTH = 16    # 2 bar of 2/4
# size of hidden layer
HIDDEN_SIZE = 64
# number of epochs to train the model
NB_EPOCHS = 200

# length of music generated
GEN_LENGTH = 500

# 1/RAND_TIME next note is chosen according to probabilities distribution
# instead of using argmax
RAND_TIME = 5

################## MIDI FILE IMPORT / EXPORT ###################################

def get_notes():
    """ Get all the notes and chords from the midi files in the directory """
    file2notes = {}
    try:
        # try to load notes data pre-extracted from midi files
        print("Load midi data from %s" % NOTES_FILE)
        with open(NOTES_FILE, 'rb') as filepath:
            file2notes = pickle.load(filepath)
    except:
        print("Fail !")
        # load data from midi files and extract notes indormations
        print("Import data from midi files found in the \"%s/\" directory instead" % MIDI_CORPUS_DIRECTORY)
        for file in glob.glob(MIDI_CORPUS_DIRECTORY + "/*.mid"):
            notes = []
            print("Parsing %s" % file)

            midi = m21.converter.parse(file)
            # all file infos, notes and chords in a flat structure
            midiflat = midi.flat
            for x in midiflat.notes:
                # a note: get offset (start time), pitch and duration
                if isinstance(x, m21.note.Note):
                    notes.append([x.offset, x.pitch.nameWithOctave, x.duration.quarterLength])
                # decompose a chord in several notes
                elif isinstance(x, m21.chord.Chord):
                    notes += [[x.offset, p.nameWithOctave, x.duration.quarterLength] for p in x.pitches]
                # # debug: show all notes
                # for n in notes:
                #     print("offset:%s note:%s duration:%s" % tuple(n))
            print("-------> done")
            file2notes[file] = notes

        print("Save data for later use")
        with open(NOTES_FILE, 'wb') as filepath:
            pickle.dump(file2notes, filepath)
    return file2notes

def notes_to_dict():
    """
    At each offset (each timestep), group all notes played together :
    it becomes an element of vocabulary. Get all vocabulary elements in the corpus.
    Arguments:
    file2notes -- a dictionary of each file name and the corresponding list of
                  notes (offset, pitch, duration)
    Returns:
    vocab -- a vector of all elements
    file2elmt -- file2notes with the notes grouped for each timestep and offset information removed.
    """
    vocab = []
    file2elmt = {}
    try:
        # try to use pre-computed data to train the network
        print("Load vocabulary and music element indexes from %s" % VOCAB_FILE)
        with open(VOCAB_FILE, 'rb') as filepath:
            (vocab, file2elmt) = pickle.load(filepath)
    except:
        print("Fail !")
        file2notes = get_notes()
        # convert notes data into one_hot vector sequences to train the network
        print("Convert music into lists of indexes and extract the vocabulary")
        vocab2idx = {}
        idx = 0
        for key, val in file2notes.items():
            new_val = []
            elmts = []
            offset = 0
            for note in val:
                # ensure that note offset is a multiple of OFFSET_STEP
                note_offset = note[0] - (note[0] % OFFSET_STEP)
                while note_offset != offset:
                    try:
                        new_val.append(vocab2idx[tuple(elmts)]) # elmt exist
                    except:
                        vocab.append(tuple(elmts))      # else create it
                        vocab2idx[tuple(elmts)] = idx   # give it an idx
                        new_val.append(idx)
                        idx += 1
                    elmts = []
                    offset += OFFSET_STEP
                elmts.append(tuple(note[1:]))
            while not new_val[0]: new_val = new_val[1:]
            while not new_val[-1]: new_val = new_val[:-1]
            file2elmt[key] = new_val

        print("Save vocabulary for later use")
        with open(VOCAB_FILE, 'wb') as filepath:
            pickle.dump((vocab, file2elmt), filepath)
    return vocab, file2elmt

def element2onehot(elmt, n_vocab):
    v = [0 for i in range(n_vocab)]
    try: v[elmt] = 1
    except: pass
    return v

def prepare_sequences(file2elmt, n_vocab):
    """ convert each sequence of notes into matrix X and Y
        each timestep is a binary vector of len(note2idx) features
        a bit is one if the note/duration is played at timestep t
        each offset is considered a multiple of OFFSET_STEP (or is rounded)
    Arguments:
    file2elmt -- file2notes with the notes grouped for each timestep and offset information removed.
    n_vocab -- number of uniq elements in the vocabulary.

    Returns:
    X -- network inputs (n_sequences, SEQ_LENGTH, n_vocab)
    Y -- network outputs (SEQ_LENGTH, n_sequences, n_vocab) shifted by one
    """
    X = []
    Y = []
    for music in file2elmt.values():
        # prepare a matrix of one_hot vectors
        music_vecs = [element2onehot(elmt, n_vocab) for elmt in music]
        # cut the music into sequences
        for i in range(len(music_vecs) - (SEQ_LENGTH+1)):
            X.append(music_vecs[i:i + SEQ_LENGTH])
            Y.append(music_vecs[i+1:i + SEQ_LENGTH+1])

    Y = np.swapaxes(Y,0,1)
    return np.array(X),np.array(Y)

def create_midi_stream(prediction_output, vocab):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for elmt in prediction_output:
        notes = vocab[elmt]
        if len(notes) != 0:
            # a chord
            if len(notes) > 1:
                notes_in_chord = []
                for current_note in notes:
                    pitch, duration = current_note
                    new_note = m21.note.Note(pitch, quarterLength=duration)
                    new_note.storedInstrument = m21.instrument.Piano()
                    notes_in_chord.append(new_note)
                new_chord = m21.chord.Chord(notes_in_chord)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # a note
            else:
                pitch, duration = notes[0]
                new_note = m21.note.Note(pitch, quarterLength=duration)
                new_note.offset = offset
                new_note.storedInstrument = m21.instrument.Piano()
                output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += OFFSET_STEP
    midi_stream = m21.stream.Stream(output_notes)

    return midi_stream

################## NETWORK MODEL TO LEARN AND GENERATE MUSIC ###################

def create_model(n_vocab):
    """
    create the structure of the neural network
    """

    reshapor = Reshape((1, n_vocab))
    LSTM_cell = LSTM(HIDDEN_SIZE, return_state = True)
    densor = Dense(n_vocab, activation='softmax')

    # Define the input of your model with a shape
    X = Input(shape=(SEQ_LENGTH, n_vocab))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(HIDDEN_SIZE,), name='a0')
    c0 = Input(shape=(HIDDEN_SIZE,), name='c0')
    a = a0
    c = c0

    # Create empty list to append the outputs while iterate
    outputs = []
    for t in range(SEQ_LENGTH):
        # Select the "t"th time step vector from X.
        x = Lambda(lambda x: X[:,t,:])(X)
        # Use reshapor to reshape x to be (1, n_vocab)
        x = reshapor(x)
        #x = Lambda(lambda x:K.print_tensor(x, message='x = '))(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        #out = Lambda(lambda x:K.print_tensor(x, message='out = '))(out)
        # Add the output to "outputs"
        outputs.append(out)

    # Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)

    # Compile the model to be trained.
    # We will use Adam and a categorical cross-entropy loss.
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, densor, LSTM_cell


def train(model, X, Y):
    """ compile and train the neural network """

    # Lets initialize `a0` and `c0` for the LSTM's init state to be 0
    m = X.shape[0] # size of training set
    a0 = np.zeros((m, HIDDEN_SIZE))
    c0 = np.zeros((m, HIDDEN_SIZE))
    # # Lets create a checkpoint to save the trained model each time it makes
    # # progress (loss decrease)
    checkpoint = ModelCheckpoint(
        WEIGHT_FILE_FMT,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    # Lets now fit the model! We will turn `Y` to a list before doing so,
    # since the cost function expects `Y` to be provided in this format
    # (one list item per time-step). So `list(Y)` is a list with SEQ_LENGTH
    # items, where each of the list items is of shape (m,n_vocab).
    # Lets train for NB_EPOCHS epochs. This will take a few minutes.
    return model.fit([X, a0, c0], list(Y), epochs=NB_EPOCHS, batch_size=64, callbacks=[checkpoint], shuffle=True)

def one_hot(x, n_vocab):
    # choose the index with highest probability
    idx = K.argmax(x)
    # TODO: Sample the index of an element within the vocabulary from the
    # probability distribution x ?
    x = tf.one_hot(idx, n_vocab)
    x = RepeatVector(1)(x)
    return x

def music_inference_model(LSTM_cell, densor, n_vocab, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_vocab -- integer, number of unique values
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """

    # Define the input of your model with a shape
    x0 = Input(shape=(1, n_vocab))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(HIDDEN_SIZE,), name='a0')
    c0 = Input(shape=(HIDDEN_SIZE,), name='c0')
    a = a0
    c = c0
    x = x0

    # Create an empty list of "outputs" to later store your predicted values
    outputs = []
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)

        # Append the prediction "out" to "outputs". out.shape = (None, 78)
        outputs.append(out)

        # Select the next value according to "out", and set "x" to be the
        # representation of the selected value, which will be passed as the
        # input to LSTM_cell on the next step
        x = Lambda(lambda x: one_hot(x, n_vocab))(out)

    # Create model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


def predict_and_sample(inference_model, x_init, a_init, c_init):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, SEQ_LENGTH),
                     one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a),
                     initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a),
                     initializing the cell state of the LSTM_cel

    Returns:
    idx -- numpy-array of shape (Ty, 1),
           matrix of indices representing the values generated
    """

    # Use your inference model to predict an output sequence
    pred = inference_model.predict([x_init, a_init, c_init])
    # Convert "pred" into an np.array() of indices with the maximum proba
    idx = np.array(np.argmax(pred, axis=-1))
    idx = [i[0] for i in idx]
    return idx

def predict_and_sample2(LSTM_cell, densor, n_vocab, x_init, Ty = 100):
    """
    Predicts the next value of values using the inference model
    and use the inference model to predict an output sequence
    """
    x0 = Input(shape=(1, n_vocab))
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(HIDDEN_SIZE,), name='a0')
    c0 = Input(shape=(HIDDEN_SIZE,), name='c0')
    a = a0
    c = c0
    x = x0
    # Perform one step of LSTM_cell
    a, _, c = LSTM_cell(x, initial_state=[a, c])
    # Apply Dense layer to the hidden state output of the LSTM_cell
    output = densor(a)
    # Create model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=output)
    # Use your inference model to predict an output sequence
    indexes = []
    x = x_init
    a_init = np.zeros((1, HIDDEN_SIZE))
    c_init = np.zeros((1, HIDDEN_SIZE))
    for t in range(Ty):
        pred = inference_model.predict([x, a_init, c_init])
        # Convert "pred" into an np.array() of indices with the maximum proba
        res = np.array(pred)[0] # first note of a sequence of 1 note
        choice = np.random.choice(range(n_vocab), p = res.ravel())
        r = np.random.randint(0,RAND_TIME)
        print(choice, np.argmax(res), r != 0)
        if r != 0:
            choice = np.argmax(res)
        indexes.append(choice)
        x = np.zeros((1, 1, n_vocab))
        x[0][0][choice] = 1

    return indexes

# https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_history(history):
    loss_list = [s for s in history.history.keys() \
                 if 'loss' in s]
    # check loss
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
    # As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' \
                 + str(format(history.history[l][-1],'.5f')) + ')')
    # legend / axis
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

################## MAIN ########################################################

if __name__ == '__main__':
    if len(argv) < 2:
        print("usage: %s [train|generate]" % argv[0])
    else:
        if argv[1] == "train":
            (vocab, file2elmt) = notes_to_dict()
            n_vocab = len(vocab)
            print("Convert data into onehot vector sequences to train the lstm network")
            (X,Y) = prepare_sequences(file2elmt, n_vocab)

            # # debug: show the 10 first element of vocabulary
            # for i in range(10):
            #     print("element %d : %s" % (i, vocab[i]))

            # # debug : retrieve elements index from onehot vectors and show the music
            # print("Example of sequence:")
            # seq = np.array(np.argmax(X[X.shape[0]/4,:,:], axis=-1))
            # create_midi_stream(seq, vocab).show()
            # raw_input("Press Enter to continue...")

            # information about inputs and outputs of the network
            print('Shape of X:', X.shape)
            print('Number of training examples:', X.shape[0])
            print('Length of sequence (Tx = X.shape[1]):', SEQ_LENGTH)
            print('Total # of unique values (n_vocab = X.shape[2]):', n_vocab)
            print('Shape of Y:', Y.shape)

            # create the network model
            print("Create the lstm network")
            (model, densor, LSTM_cell) = create_model(n_vocab)
            try:
                # try to load pre-trained weight from a previous IDENTICAL model
                # (using same data)
                print("Load weights from a pretrained model: %s" % WEIGHT_FILE)
                model.load_weights(WEIGHT_FILE)
            except:
                print("Fail !")

            print("Train the lstm network for %d epochs" % NB_EPOCHS)
            history = train(model, X, Y)
            with open(LAYERS_FILE, 'wb') as filepath:
                pickle.dump((densor, LSTM_cell), filepath)
            print("Plot loss history")
            plot_history(history)
        else:
            vocab = []
            n_vocab = 0
            densor = None
            LSTM_cell = None
            try:
                print("Load vocabulary and music element indexes from %s" % VOCAB_FILE)
                with open(VOCAB_FILE, 'rb') as filepath:
                    (vocab, _) = pickle.load(filepath)
                n_vocab = len(vocab)
                print("Create a model to generate some music")
                with open(LAYERS_FILE, 'rb') as filepath:
                    (densor, LSTM_cell) = pickle.load(filepath)
                inf_model = music_inference_model(LSTM_cell, densor, n_vocab, GEN_LENGTH)
                print("Load weights from the pretrained model: %s" % WEIGHT_FILE)
                model.load_weights(WEIGHT_FILE)
            except:
                print("Missing informations... try to train the network first")

            print("Generate some music")
            x_init = np.zeros((1, 1, n_vocab))
            a_init = np.zeros((1, HIDDEN_SIZE))
            c_init = np.zeros((1, HIDDEN_SIZE))
            idx = predict_and_sample(inf_model, x_init, a_init, c_init)
            midi_stream = create_midi_stream(idx, vocab)
            midi_stream.write('midi', fp='midi_test_output.mid')

            idx = predict_and_sample2(LSTM_cell, densor, n_vocab, x_init, GEN_LENGTH)
            midi_stream = create_midi_stream(idx, vocab)
            midi_stream.write('midi', fp='midi_test_output4.mid')

            x_init = np.zeros((1, 1, n_vocab))
            x_init[0][0][len(vocab)/2] = 1
            a_init = np.zeros((1, HIDDEN_SIZE))
            c_init = np.zeros((1, HIDDEN_SIZE))
            idx = predict_and_sample(inf_model, x_init, a_init, c_init)
            midi_stream = create_midi_stream(idx, vocab)
            midi_stream.write('midi', fp='midi_test_output2.mid')

            x_init = np.zeros((1, 1, n_vocab))
            x_init[0][0][len(vocab)/4] = 1
            a_init = np.zeros((1, HIDDEN_SIZE))
            c_init = np.zeros((1, HIDDEN_SIZE))
            idx = predict_and_sample(inf_model, x_init, a_init, c_init)
            midi_stream = create_midi_stream(idx, vocab)
            midi_stream.write('midi', fp='midi_test_output3.mid')
