# Generation de ragtime avec un réseau LSTM dans Keras

## Sources de documentation utilisées

* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy)
* [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5) (Sigurður Skúli)
* [Deep Learning Techniques for Music Generation - A Survey](https://www.researchgate.net/publication/319524552_Deep_Learning_Techniques_for_Music_Generation_-_A_Survey) (Jean-Pierre BriotGaëtan HadjeresFrancois PachetFrancois Pachet)
* [Coursera MOOC on Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/) by Andrew Ng

## Choix d'un modèle

Sur sa page, Igurður Skúli propose de donner toute une séquence de notes à un réseau LSTM pour prédire la note suivante (modèle many to one tel que décrit par Andrej Karpathy) tandis que Andrew Ng propose dans son cours un modèle qui prédit la note suivante pour chaque note de la séquence (modèle many to many).

Les exemples de musiques obtenues par Igurður Skúli semblent intéressants, cependant après 4 jours d'apprentissage (200 epochs) sur le même corpus, je constate que les échantillons de musique générés ne sont pas tous aussi convainquant. J'ai testé le modèle sur un corpus de quelques morceaux de ragtime tiré du site [http://www.trachtman.org/ragtime/](spacer 	Ragtime Piano MIDI files by Warren Trachtman) et auxquels j'ai fait subir les prétraitements requis. Le modèle ne converge plus du tout, ce qui laisse penser qu'il est très dépendant de la structure des données utilisées pour l'apprentissage.

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

## Préparation des données

Dans des pièces pour piano, plusieurs notes peuvent être jouées ensembles à chaque temps `t`. Igurður Skúli comme Andrew Ng, proposent de créer un dictionnaire de toutes les notes ou accords pouvant apparaître dans leur corpus à chaque temps `t`. Chaque élément (note ou accord) est alors représenté par un index qui constitue une catégorie. Leur réseau est donc chargé de résoudre un problème de régression logistique qui consiste à prédire une catégorie en sortie (note ou accord suivant) en fonction d'une autre catégorie en entrée (note ou accord courant).

J'ai ici fait le choix de créer moi aussi une catégorie unique pour chaque élément (ensemble de note) pouvant être joué à un instant t. Pour fournir cette donnée au réseau, elle est représentée par un vecteur binaire dont tous les bits sont à 0 sauf un dont la valeur 1 représente la catégorie (one-hot-encoding).
Une autre approche à tester est de créer une catégorie par note et de coder une donnée à un instant t par un vecteur dont plusieurs bits sont à 1, un pour chaque note jouée (multi-hot-encoding). cf. [Guide to multi-class multi-label classification with neural networks in python](https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/).

### Importation des données

J'ai utilisé comme corpus d'apprentissage pour mon réseau l'ensemble des fichiers MIDI de ragtime de Scott Joplin proposés sur le site [Ragtime Piano MIDI files by Warren Trachtman](http://www.trachtman.org/ragtime/). Le programme va chercher les fichiers MIDI dans le répertoire désigné par `MIDI_CORPUS_DIRECTORY`.

Pour traiter les fichiers MIDI, j'ai utilisé le module [http://web.mit.edu/music21/](music21) de Python. Chaque fichier MIDI est parsé et toutes les pistes (deux pistes pour le piano) sont "aplaties" en un seul ensemble d'informations contenant toutes les notes et accords du morceau.
```
        midi = m21.converter.parse(file) # parse
        midiflat = midi.flat             # all infos in a flat structure
```
Pour chaque note, je récupère sa hauteur (`nameWithOctave`), sa durée par rapport à une noire (`duration.quarterLength`, croche=0.5, double=0.25, etc...) et l'instant ou la note est jouée (`offset`). Pour les accords, je récupère la hauteur des différentes notes jouées (`pitches`) ainsi que la durée et l'offset. Je récupère ainsi un ensemble de notes.
La fonction `get_notes(directory)` retourne donc un dictionnaire `file2notes`faisant correspondre à chaque fichier MIDI extrait du répertoire `directory` la liste des notes sous la forme [offset, hauteur, durée].

Si je décompose les accords en notes, c'est dans l'idée de pouvoir tester plus tard le _multi-hot-encoding_.

### Extraction du vocabulaire et conversion en listes d'index

A chaque temps `t` ou offset, plusieurs notes peuvent être jouées ensembles (accord + note de la mélodie). Comme le suggère Sigurður Skúli, j'inspecte les notes extraites des fichiers MIDI pour identifier l’intervalle de temps le plus courant entre les notes de mon corpus. Cet intervalle est 0.25, ce qui correspond à une double croche. Je partitionne donc la liste des notes en fonction de chaque offset par pas de 0.25. Le pas est stocké dans la variable globale `OFFSET_STEP`, ce qui permet facilement de la modifier en fonction du corpus. Le cas échéant, les notes dont l'offset n'est pas multiple de OFFSET_STEP sont "décalées" dans le temps. Ces modification sont négligeables et n'altèrent pas la musique de manière significative.
Chacun des ensembles de notes extraits constitue un "élément" du vocabulaire disponible pour générer de la musique. Comme les musiques peuvent contenir des silences, l'ensemble vide fait également partie du vocabulaire.

La fonction  `notes_to_dict(file2notes)` récupère chacun de ces élément de vocabulaire dans une liste et transforme le dictionnaire `file2notes` précédent en faisant correspondre à chaque fichier une liste d'index qui représentent les éléments de la musique. La fonction retourne le vocabulaire `vocab` et le dictionnaire modifié `file2elmt`.

Pour éviter de recalculer ces éléments à chaque fois, `file2notes` est enregistré dans le fichier désigné par NOTES_FILE, et `vocab` et `file2elmt`sont enregistré dans le fichier désigné par VOCAB_FILE grâce au module [pickle](https://docs.python.org/3/library/pickle.html) de Python.

### Préparation des séquences pour entraîner le réseau

Les différents morceaux de musique représentés par des listes d'index sont découpés en séquences de notes dont la longueur est définie par `SEQ_LENGTH`.
Après quelques tests, j'ai pu constater qu'entraîner le réseau avec des séquences longues (SEQ_LENGTH=32 équivalent à 4 mesures à 2 temps) encourage le réseau à générer ensuite de longues séquences "plagiées" dans le corpus. Tandis que des séquences plus courtes (SEQ_LENGTH=8 équivalent à 1 seule mesure) permettent de diminuer cet effet.

La fonction `prepare_sequences(file2elmt, n_vocab)` reçoit le dictionnaire `file2elmt` ainsi que la taille du vocabulaire. Elle découpe chaque liste d'index en séquences et chaque index d'une séquence est converti en vecteur one-hot. La fonction retourne l'ensemble des séquences encodées sous forme de vecteurs (variable X) et les sorties attendues qui correspondent aux mêmes séquences décalées d'un temps `t`.

Les séquences produites ici se recoupent. Dans l'exemple proposé par Andrew Ng dans son cours, les séquences utilisées pour entraîner le réseau sont tirées au hasard dans l'ensemble du corpus. Ne pas utiliser toutes les séquences doit permettre d'éviter que le réseau ne sur-apprenne les morceaux du corpus et ne produise trop de plagiat. Igurður Skúli de son côté utilise toutes les séquences, mais son réseau possède différents couches dropout qui doivent permettre d'éviter également le problème de sur-apprentissage.



