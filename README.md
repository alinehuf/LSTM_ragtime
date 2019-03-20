# Generation de ragtime avec un réseau LSTM dans Keras

## Sources de documentation utilisées

* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy)
* [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5) (Sigurður Skúli)
* [Deep Learning Techniques for Music Generation - A Survey](https://www.researchgate.net/publication/319524552_Deep_Learning_Techniques_for_Music_Generation_-_A_Survey) (Jean-Pierre BriotGaëtan HadjeresFrancois PachetFrancois Pachet)
* [Coursera MOOC on Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/) by Andrew Ng

## Choix d'un modèle

Sur sa page, Igurður Skúli propose de donner toute une séquence de notes à un réseau LSTM pour prédire la note suivante (modèle many to one tel que décrit par Andrej Karpathy) tandis que Andrew Ng propose dans son cours un modèle qui prédit la note suivante pour chaque note de la séquence (modèle many to many).

Les exemples de musiques obtenues par Igurður Skúli semblent intéressants, cependant après 4 jours d'apprentissage (200 epochs) sur le même corpus, je constate que les échantillons de musique générés ne sont pas tous aussi convainquant. J'ai testé le modèle sur un corpus de quelques morceaux de ragtime tiré du site [Ragtime Piano MIDI files by Warren Trachtman](http://www.trachtman.org/ragtime/) et auxquels j'ai fait subir les prétraitements requis. Le modèle ne converge plus du tout, ce qui laisse penser qu'il est très dépendant de la structure des données utilisées pour l'apprentissage.

Le modèle proposé dans le cours de Andrew Ng étant prévu pour un usage pédagogique, il présente l'avantage d'être plus simple, plus rapide à entraîner et les résultats sont plus facilement reproductibles. J'ai donc choisi de partir de ce modèle:

```
      ŷ<t>
       ↑
   ┌───────┐
   │softmax│
   └───────┘
       ↑
     ┌────┐ a<t>, c<t>
     │    │ ----┐
     │LSTM│     ■ (transition à t+1)
     │    │ <---┘
     └────┘
       ↑
      x<t>
```
`x<t>` représente la donnée d'entrée au temps `t`, `ŷ<t>` est la sortie prédite par le réseau. `a<t>` correspond à l'activation de la cellule LSTM ré-injectée au temps `t+1` et `c<t>` correspond au contenu de la cellule mémoire permettant d'apprendre des dépendances à long terme à l'intérieur d'une séquence.

Si on déplie le modèle, à chaque temps `t`, il doit prédire la note suivante `x<t+1>`. La musique est découpée en "notes" et le modèle est entraîné à l'aide d'un ensemble de séquences extraites du corpus.

```
attendu:  x<t+1>       x<t+2>       x<t+3>       x<t+4>       x<t+5>
prédit:   ŷ<t+1>       ŷ<t+2>       ŷ<t+3>       ŷ<t+4>       ŷ<t+5>
            ↑            ↑            ↑            ↑            ↑
        ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
        │softmax│    │softmax│    │softmax│    │softmax│    │softmax│
        └───────┘    └───────┘    └───────┘    └───────┘    └───────┘
            ↑            ↑            ↑            ↑            ↑
          ┌────┐ a<t>  ┌────┐ a<t>  ┌────┐ a<t>  ┌────┐ a<t>  ┌────┐
          │    │ ----> │    │ ----> │    │ ----> │    │ ----> │    │   ...
          │LSTM│ c<t>  │LSTM│ c<t>  │LSTM│ c<t>  │LSTM│ c<t>  │LSTM│
          │    │ ----> │    │ ----> │    │ ----> │    │ ----> │    │
          └────┘       └────┘       └────┘       └────┘       └────┘
            ↑            ↑            ↑            ↑            ↑
           x<t>        x<t+1>       x<t+2>       x<t+3>       x<t+4>
```
Pour générer une séquence de notes, une première note est proposée au modèle au temps `t`, la prédiction produite par le réseau est ensuite ré-injectée en entrée comme source pour produire la note suivante.

```
      ŷ<t> --------------------------┐
       ↑                             |
   ┌───────┐                         |
   │softmax│                         |
   └───────┘                         |
       ↑                             |
     ┌────┐ a<t>, c<t>               |
     │    │ ----┐                    |
     │LSTM│     ■ (transition à t+1) ■
     │    │ <---┘                    |
     └────┘                          |
       ↑                             |
      x<t>  <------------------------┘
```

## Préparation des données (Import/Export)

Dans des pièces pour piano, plusieurs notes peuvent être jouées ensembles à chaque temps `t`. Igurður Skúli comme Andrew Ng, proposent de créer un dictionnaire de toutes les notes ou accords pouvant apparaître dans leur corpus à chaque temps `t`. Chaque élément (note ou accord) est alors représenté par un index qui constitue une catégorie. Leur réseau est donc chargé de résoudre un problème de régression logistique qui consiste à prédire une catégorie en sortie (note ou accord suivant) en fonction d'une autre catégorie en entrée (note ou accord courant).

J'ai fait le choix de créer moi aussi une catégorie unique pour chaque élément (ensemble de note) pouvant être joué à un instant t. Pour fournir cette donnée au réseau, elle est représentée par un vecteur binaire dont tous les bits sont à 0 sauf un dont la valeur 1 représente la catégorie (one-hot-encoding).
Une autre approche à tester est de créer une catégorie par note et de coder une donnée à un instant t par un vecteur dont plusieurs bits sont à 1, un pour chaque note jouée (multi-hot-encoding). cf. [Guide to multi-class multi-label classification with neural networks in python](https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/).

### Importation des données

J'ai utilisé comme corpus d'apprentissage pour mon réseau l'ensemble des fichiers MIDI de ragtime de Scott Joplin proposés sur le site [Ragtime Piano MIDI files by Warren Trachtman](http://www.trachtman.org/ragtime/). Le programme va chercher les fichiers MIDI dans le répertoire désigné par `MIDI_CORPUS_DIRECTORY`.

Pour traiter les fichiers MIDI, j'ai utilisé le module [http://web.mit.edu/music21/](music21) de Python. Chaque fichier MIDI est parsé et toutes les pistes (deux pistes pour le piano) sont "aplaties" en un seul ensemble d'informations contenant toutes les notes et accords du morceau.
```
        midi = m21.converter.parse(file) # parse
        midiflat = midi.flat             # all infos in a flat structure
```
Pour chaque note, je récupère sa hauteur (`nameWithOctave`), sa durée par rapport à une noire (`duration.quarterLength`, croche=0.5, double=0.25, etc...) et l'instant ou la note est jouée (`offset`). Pour les accords, je récupère la hauteur des différentes notes jouées (`pitches`) ainsi que la durée et l'offset. Je récupère ainsi un ensemble de notes.
La fonction `get_notes()` retourne donc un dictionnaire `file2notes`faisant correspondre à chaque fichier MIDI extrait du répertoire `MIDI_CORPUS_DIRECTORY`la liste des notes sous la forme [offset, hauteur, durée].

Si je décompose les accords en notes, c'est dans l'idée de pouvoir tester plus tard le _multi-hot-encoding_.

### Extraction du vocabulaire et conversion en listes d'index

A chaque temps `t` ou offset, plusieurs notes peuvent être jouées ensembles (accord + note de la mélodie). Comme le suggère Sigurður Skúli, j'inspecte les notes extraites des fichiers MIDI pour identifier l’intervalle de temps le plus courant entre les notes de mon corpus. Cet intervalle est 0.25, ce qui correspond à une double croche. Je partitionne donc la liste des notes en fonction de chaque offset par pas de 0.25. Le pas est stocké dans la variable globale `OFFSET_STEP`, ce qui permet facilement de la modifier en fonction du corpus. Le cas échéant, les notes dont l'offset n'est pas multiple de OFFSET_STEP sont "décalées" dans le temps. Ces modification sont négligeables et n'altèrent pas la musique de manière significative.
Chacun des ensembles de notes extraits constitue un "élément" du vocabulaire disponible pour générer de la musique. Comme les musiques peuvent contenir des silences, l'ensemble vide fait également partie du vocabulaire.

La fonction  `notes_to_dict()` récupère chacun de ces élément de vocabulaire dans une liste et transforme le dictionnaire `file2notes` précédent en faisant correspondre à chaque fichier une liste d'index qui représentent les éléments de la musique. La fonction retourne le vocabulaire `vocab` et le dictionnaire modifié `file2elmt`.

Pour éviter de recalculer ces éléments à chaque fois, `file2notes` est enregistré dans le fichier désigné par NOTES_FILE, et `vocab` et `file2elmt`sont enregistré dans le fichier désigné par VOCAB_FILE grâce au module [pickle](https://docs.python.org/3/library/pickle.html) de Python.

### Préparation des séquences pour entraîner le réseau

Les différents morceaux de musique représentés par des listes d'index sont découpés en séquences de notes dont la longueur est définie par `SEQ_LENGTH`.
Après quelques tests, j'ai pu constater qu'entraîner le réseau avec des séquences longues (SEQ_LENGTH=32 équivalent à 4 mesures à 2 temps) encourage le réseau à générer ensuite de longues séquences "plagiées" dans le corpus. Tandis que des séquences plus courtes (SEQ_LENGTH=8 équivalent à 1 seule mesure) permettent de diminuer cet effet.

La fonction `prepare_sequences(file2elmt, n_vocab)` reçoit le dictionnaire `file2elmt` ainsi que la taille du vocabulaire. Elle découpe chaque liste d'index en séquences et chaque index d'une séquence est converti en vecteur one-hot. La fonction retourne l'ensemble des séquences encodées sous forme de vecteurs (variable X) et les sorties attendues qui correspondent aux mêmes séquences décalées d'un temps `t`.

Les séquences produites ici se recoupent. Dans l'exemple proposé par Andrew Ng dans son cours, les séquences utilisées pour entraîner le réseau sont tirées au hasard dans l'ensemble du corpus. Ne pas utiliser toutes les séquences doit permettre d'éviter que le réseau ne sur-apprenne les morceaux du corpus et ne produise trop de plagiat. Igurður Skúli de son côté utilise toutes les séquences, mais son réseau possède différents couches dropout qui doivent permettre d'éviter également le problème de sur-apprentissage.

### Conversion inverse: liste d'index vers fichier MIDI

La fonction `create_midi_stream(prediction_output, vocab)` permet de faire la conversion inverse. A partir d'une liste d'index (`prediction_output`) générée par le réseau et du vocabulaire utilisé (`vocab`), elle reconstitue un ensemble de notes et retourne un flux MIDI (`music21.stream.Stream`) qu'il est ensuite possible d'enregistrer dans un fichier.

## Création du modèle pour l'apprentissage

La fonction `create_model(n_vocab)` permet de créer le modèle d'après l'exemple donné dans le cours de Andrew Ng. Pour récupérer la sortie (prédiction) à chaque temps t, le réseau est créé avec une boucle pour itérer sur chaque élément de la séquence.

Les couches LSTM et_softmax_(Dense) sont créés une fois pour toutes pour être réutilisées à chaque étape:
```
    LSTM_cell = LSTM(HIDDEN_SIZE, return_state = True)
    densor = Dense(n_vocab, activation='softmax')
```
`HIDDEN_SIZE` est une variable globale définissant le nombre de neurones dans la couche cachée. Ces deux couches sont enregistrées dans le fichier `LAYERS_FILE`, à l'aide du module pickle de Python, afin d'être réutilisées plus tard dans le modèle d'inférence utilisé pour générer la musique.

A chaque étape, l'élément au temps `t` est extrait de l'ensemble des données d'entraînement X et remit au bon format avec `reshapor = Reshape((1, n_vocab))`.
Une itération de la cellule LSTM est calculée et son activation `a` est ajoutée à la liste des sorties `outputs`.
Le modèle utilise une optimisation _Adam_ qui associe _momentum_ (évite les oscillations des poids du réseau et accélère la descente de gradient) et _RMSprop_ (augmente la descente dans le "bon" sens et la réduit dans le sens des oscillations). Le coût est évalué avec `categorical_crossentropy`, choix classique dans le cas d'un problème de régression logistique multi-catégories.

```
    outputs = []
    for t in range(SEQ_LENGTH):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

## Entraînement du modèle

La fonction `train(model, X, Y)` initialise `a0` et `c0` avec des vecteurs à 0 et entraîne le réseau avec les données préparée dans `X` et `Y`. Afin de ne pas perdre le résultat de l'entraînement s'il est nécessaire d’interrompre les calculs, comme Sigurður Skúli, j'ai utilisé un `ModelCheckpoint` qui permet de sauvegarder les poids du modèle à la fin de chaque "epoch" (quand la performance du réseau s'améliore).

Le modèle est entraîné pendant un nombre d'époques défini dans `NB_EPOCHS`.
Les données sont découpées en mini-batch (`batch_size=64`) et sont mélangées (`shuffle=True`) pour éviter que la similarité des séquences successives n'aient un effet néfaste sur l'apprentissage.

## Création du modèle pour l'inférence

Pour générer la musique, il est nécessaire de modifier légèrement le modèle. Cette fois, au temps `t+1`, ce n'est pas le "vrai" élément suivant `x<t+1>` qui est donné au modèle mais le résultat de la prédiction `ŷ<t>`.

La sortie `ŷ<t>` fournie par la couche _softmax_ correspond à une distribution de probabilités sur l'ensemble des catégories. La somme des probabilités est 1. Si la prédiction était parfaite, seule la vraie catégorie obtiendrait une probabilité de 1 et toutes les autres 0. Andrew Ng propose d'extraire l'élément le plus probable avec la fonction `argmax` puis de re-soumettre cet élément en entrée du réseau, sous forme de vecteur one-hot, au temps t+1.

```
def one_hot(x, n_vocab):
    idx = K.argmax(x)             # index with max proba
    x = tf.one_hot(idx, n_vocab)  # one-hot encoding
    x = RepeatVector(1)(x)        # reshape input
    return x
```

La boucle utilisée pour créer le modèle ressemble donc à cela:
```
    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(lambda x: one_hot(x, n_vocab))(out)
```
L'entrée `x` est traitée par la couche LSTM puis l'activation `a` est passée à la couche _softmax_ pour produire la sortie `out`. la fonction `one_hot(x, n_vocab)` est appliquée à la sortie via une couche spéciale `Lambda` pour produire l'entrée `x` de l'itération suivante.

La fonction `predict_and_sample(inference_model, x_init, a_init, c_init)` permet de générer une séquence de notes avec le modèle d'inférence à partir de valeurs initiales pour `x`, `a`, et `c`.


Comme la prédiction est déterministe (même entrée, même sortie), le modèle a tendance à générer des boucles à la manière d'un disque vinyle rayé. Pour éviter cela, dans `predict_and_sample2()` j'ai créé un modèle pour l'inférence réduit à une seul temps `t`, ce qui me permet pour chaque entrée de récupérer la distribution de probabilités générée en sortie (ainsi que l'activation `a` et le contenu de la mémoire à long terme `c`). Au lieu de choisir systématiquement l'élément suivant le plus probable, si la répétition d'un motif est détectée, je choisis l'élément suivant au hasard en fonction de la distribution de probabilités. Ceci permet de "casser" une éventuelle boucle infinie. Les valeurs de `a` et `c` sont réinjectées à chaque itération pour prédire chaque note en fonction de ce qui précède.

## Résultats

Le modèle, très simple, entraîné sur un corpus assez réduit, réussit a "apprendre" la structure des pièces de ragtime. Le résultat est une sorte de pot pourri mêlant de courts extraits de différentes pièces. J'ai entraîné le modèle pendant 200 _epochs_ ce qui correspond à une dizaine d'heures de calcul.

La longueur des séquences utilisées pour l'apprentissage a une forte influence sur le résultat. Avec des séquences longues, le modèle a tendance à plagier de longues parties des pièces données pour l'apprentissage. Avec des séquences courtes, le résultat est décousu et peu musical. Des séquences équivalent à 2 mesures semblent fournir le meilleur compromis.

Pour générer la musique, il est possible de choisir chaque note de deux manières différentes en fonction du résultat de la couche _softmax_. Soit, la note la plus probable est extraite avec _argmax_, soit la note est choisie au hasard en fonction de la distribution de probabilités. La première approche présente un inconvénient notable dû au déterminisme du modèle : l’apparition de boucles à la manière d'un disque vinyle rayé. La seconde introduit une trop grande dose de hasard qui rend la musique générée peu esthétique. J'ai expérimenté un compromis qui consiste à choisir la note au hasard selon la distribution de probabilités uniquement lorsqu'une répétition est détectée, afin de briser une éventuelle boucle.

Les résultats obtenus sont disponibles dans les fichiers "midi_test_output_X.mid" pour _argmax_ et "midi_test_output_X_bis.mid" pour la version avec prévention des boucles.










