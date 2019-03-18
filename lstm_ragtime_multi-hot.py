#! /usr/bin/env python
""" This module prepares midi file data and feeds it to the neural
    network for training """

import glob
import music21 as m21
import pickle
import numpy as np
from keras.layers import Reshape, LSTM, Dense, Input, Lambda, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import matplotlib.pyplot as plt

# directory where is the midi file corpus
MIDI_CORPUS_DIRECTORY = "one_joplin"
# file name to save precomputed notes (notes extracted from midi file corpus)
NOTES_FILE = "data/notes2"
# offset variation from one note to the next (1=quarter, 0.25=16th note)
OFFSET_STEP = 0.25
# length of a music sequence
SEQ_LENGTH = 16
# size of hidden layer
HIDDEN_SIZE = 64
# number of epochs to train the model
NB_EPOCHS = 100

# threashold to consider the note played (1) or not (0)
THRESHOLD = 0.5

# length of music generated
GEN_LENGTH = 100

################## MIDI FILE IMPORT / EXPORT ###################################

def get_notes(directory):
    """ Get all the notes and chords from the midi files in the directory """
    file2notes = {}
    print(glob.glob(directory + "/*.mid"))

    for file in glob.glob(directory + "/*.mid"):
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
            # #debug: show all notes
            # for n in notes:
            #     print("offset:%s note:%s duration:%s" % tuple(n))
        print("-------> done")
        file2notes[file] = notes
    return file2notes

def notes_to_dict(file2notes):
    """ get all uniq notes/duration in the corpus, return a vector of all notes
        and a dictionnary given the index of each note """
    allnotes = []
    for music in file2notes.values():
        allnotes += [tuple(note[1:]) for note in music]
    allnotes = sorted(list(set(allnotes)))
    note2idx = { allnotes[i] : i for i in range(0, len(allnotes) ) }
    # debug: show the dictionary of all uniq notes with duration

    return allnotes, note2idx

def prepare_sequences(file2notes, note2idx):
    """ convert each sequence of notes into matrix X and Y
        each timestep is a binary vector of len(note2idx) features
        a bit is one if the note/duration is played at timestep t
        each offset is considered a multiple of OFFSET_STEP (or is rounded)"""
    X = []
    Y = []
    nul = [[0 for i in range(len(note2idx))]]
    for music in file2notes.values():
        # length of the music according to the last offset
        music_size = int(music[-1][0] / OFFSET_STEP + 1)
        # prepare a matrix of music_size timesteps,
        # notes played at offset o are represented as a binary vector
        music_vecs = [ [0 for i in range(len(note2idx))] \
                       for j in range(music_size) ]
        for note in music:
            offset = int(note[0] / OFFSET_STEP) # corrected offset
            name = tuple(note[1:])
            music_vecs[offset][note2idx[name]] = 1
        # cut the music into sequences
        for i in range(len(music_vecs) - SEQ_LENGTH):
            X.append(nul + music_vecs[i:i + SEQ_LENGTH - 1])
            Y.append(music_vecs[i:i + SEQ_LENGTH])

    Y = np.swapaxes(Y,0,1)
    return np.array(X),np.array(Y)

def create_midi_stream(prediction_output, allnotes):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for timestep in prediction_output:
        notes = np.where(np.array(timestep) > THRESHOLD)[0]
        if notes.shape[0] != 0:
            # a chord
            if notes.shape[0] > 1:
                notes_in_chord = []
                for current_note in notes:
                    pitch, duration = allnotes[current_note]
                    new_note = m21.note.Note(pitch, quarterLength=duration)
                    new_note.storedInstrument = m21.instrument.Piano()
                    notes_in_chord.append(new_note)
                new_chord = m21.chord.Chord(notes_in_chord)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # a note
            else:
                pitch, duration = allnotes[notes[0]]
                new_note = m21.note.Note(pitch, quarterLength=duration)
                new_note.offset = offset
                new_note.storedInstrument = m21.instrument.Piano()
                output_notes.append(new_note)
            print(output_notes[-1])
        # increase offset each iteration so that notes do not stack
        offset += OFFSET_STEP
    midi_stream = m21.stream.Stream(output_notes)

    return midi_stream

################## MIDI FILE IMPORT / EXPORT ###################################

def create_model(X, n_vocab):
    """
    create the structure of the neural network

    Arguments:
    X -- inputs corpus
    n_vocab -- number of unique values in the music data

    Returns:
    model -- a keras model
    """

    reshapor = Reshape((1, n_vocab))
    LSTM_cell = LSTM(HIDDEN_SIZE, return_state = True)
    densor = Dense(n_vocab, activation='sigmoid')

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
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
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
    filepath = "weight/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
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
    return model.fit([X, a0, c0], list(Y), epochs=NB_EPOCHS, batch_size=64, callbacks=[checkpoint])


def binary_vec(x):
    x = K.greater(x, THRESHOLD)
    x = K.cast(x, "float32")
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
        x = Lambda(binary_vec)(out)


    # Create model instance
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


# https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



################## MAIN ########################################################

if __name__ == '__main__':
    print("Import data from midi files found in the \"%s/\" directory" \
           % MIDI_CORPUS_DIRECTORY)
    file2notes = {}
    try:
        with open(NOTES_FILE, 'rb') as filepath:
            file2notes = pickle.load(filepath)
    except:
        file2notes = get_notes(MIDI_CORPUS_DIRECTORY)
        with open(NOTES_FILE, 'wb') as filepath:
            pickle.dump(file2notes, filepath)

    # debug
    # f = next(iter(file2notes))
    # print("Example of midi file: \"%s\"" % f)
    # m21.converter.parse(f).flat.show()
    # raw_input("Press Enter to continue...")

    print("Convert data into sequences to train the lstm network")
    (allnotes, note2idx) = notes_to_dict(file2notes)
    (X,Y) = prepare_sequences(file2notes, note2idx)
    n_vocab = len(allnotes)

    # debug
    # print("Example of sequence:")
    # create_midi_stream(X[0,:,:], allnotes).show()
    # raw_input("Press Enter to continue...")

    print('Shape of X:', X.shape)
    print('Number of training examples:', X.shape[0])
    print('Length of sequence (Tx = X.shape[1]):', SEQ_LENGTH)
    print('Total # of unique values (n_vocab = X.shape[2]):', n_vocab)
    print('Shape of Y:', Y.shape)

    print("Create the lstm network")
    (model, densor, LSTM_cell) = create_model(X, n_vocab)
    print("Train the lstm network")
    history = train(model, X, Y)
    print("Plot loss history")
    plot_history(history)

    print("Generate some music")
    inference_model = music_inference_model(LSTM_cell, densor, n_vocab, GEN_LENGTH)
    x_init = np.zeros((1, 1, n_vocab))
    # (a,b) = (np.random.randint(0, X.shape[0]), np.random.randint(0, X.shape[1]))
    # x_init[0][0][:] = X[a][b][:]
    x_init[0][0][X.shape[2]/2] = 1
    a_init = np.zeros((1, HIDDEN_SIZE))
    c_init = np.zeros((1, HIDDEN_SIZE))
    prediction_output = inference_model.predict([x_init, a_init, c_init])
    prediction_output = np.array([x_init] + [p[0] for p in prediction_output])
    print prediction_output
    midi_stream = create_midi_stream(prediction_output, allnotes)
    midi_stream.write('midi', fp='midi_test_output.mid')


