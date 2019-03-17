# Generation de ragtime avec un réseau LSTM dans Keras

## Documentation:
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy)
- How to Generate Music using a LSTM Neural Network in Keras (Sigurður Skúli)
- [Deep Learning Techniques for Music Generation - A Survey](https://www.researchgate.net/publication/319524552_Deep_Learning_Techniques_for_Music_Generation_-_A_Survey)
- Coursera MOOC on Sequence Models by Andrew Ng

## Choix d'un modèle

Sur sa page, Igurður Skúli propose de donner toute une séquence de notes à un réseau LSTM pour prédire la note suivante (modèle many to one tel que décrit par Andrej Karpathy) tandis que Andrew Ng propose dans son cours un modèle qui prédit la note suivante pour chaque note de la séquence (modèle many to many).

Les exemples de musiques obtenues par Igurður Skúli semblent intéressants, cependant après 4 jours d'apprentissage (200 epochs) sur le même corpus, je constate que les échantillons de musique générés ne sont pas tous aussi convainquant. J'ai testé le modèle sur un corpus de quelques morceaux de ragtime tiré du site [http://www.trachtman.org/ragtime/](spacer 	Ragtime Piano MIDI files by Warren Trachtman) et auxquels j'ai fait subir les prétraitements requis. Le modèle ne converge plus du tout, ce qui laisse penser qu'il est très dépendant de la structure des données utilisées pour l'apprentissage.

Le modèle proposé dans le cours de Andrew Ng étant prévu pour un usage pédagogique, il présente l'avantage d'être plus simple, plus rapide à entraîner et les résultats sont plus facilement reproductibles. J'ai donc choisi de partir de ce modèle.

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


