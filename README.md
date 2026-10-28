from pathlib import Path
import pypandoc

content = r"""# 🍕 Food Vision — Classification d’images avec un CNN

## 1. Présentation du projet

Ce projet de Deep Learning utilise un réseau de neurones convolutif (**CNN — Convolutional Neural Network**) pour classifier des images de nourriture en **10 catégories**. Le notebook `Projet_Deep_Learning.ipynb` couvre les étapes principales d’un projet de vision par ordinateur : récupération des données, exploration, prétraitement, création du modèle, entraînement et prédiction sur une nouvelle image.

## 2. Objectif

Construire et entraîner un modèle capable de recevoir une image de nourriture et de prédire la catégorie à laquelle elle appartient, parmi les classes présentes dans le jeu de données.

## 3. Technologies utilisées

- **Python** : langage de programmation.
- **Jupyter Notebook / VS Code** : environnement d’exécution du notebook.
- **TensorFlow / Keras** : construction et entraînement du réseau de neurones.
- **NumPy** : manipulation des tableaux et des noms de classes.
- **Matplotlib** : visualisation des images et des courbes.
- **Pandas** : organisation et visualisation de l’historique d’entraînement.
- **`urllib`, `zipfile`, `os`, `pathlib`, `random`** : téléchargement, extraction et exploration des fichiers.

## 4. Jeu de données

Le notebook télécharge l’archive suivante :

<https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip>

Après extraction, il s’attend à trouver cette structure :

```text
10_food_classes_all_data/
├── train/
│   ├── classe_1/
│   ├── classe_2/
│   └── ...
└── test/
    ├── classe_1/
    ├── classe_2/
    └── ...
```

Chaque sous-dossier correspond à une classe et contient ses images. Les noms réels des classes sont lus automatiquement à partir des sous-dossiers de `train/`. Le notebook ne fixe donc pas manuellement la liste des dix catégories.

## 5. Étapes du projet

### Étape 1 — Télécharger et extraire les données

Le notebook récupère l’archive ZIP puis l’extrait dans le répertoire courant. Il parcourt ensuite les dossiers pour afficher le nombre de sous-dossiers et d’images.

### Étape 2 — Explorer et visualiser les images

Le code récupère les noms des classes, affiche plusieurs images d’une catégorie choisie (par exemple `pizza`) et charge également une image de la catégorie `ice_cream`. Cette étape permet de vérifier visuellement les données.

### Étape 3 — Prétraiter les images

Les images sont préparées avec `ImageDataGenerator` :

- **Normalisation** : les valeurs des pixels sont ramenées approximativement entre 0 et 1 avec `rescale=1/255`.
- **Redimensionnement** : toutes les images sont ramenées à `224 × 224` pixels.
- **Batchs** : les images sont chargées par groupes de 32.
- **Augmentation des données d’entraînement** : rotations, décalages horizontaux et verticaux, zoom et retournement horizontal.
- **Données de test** : seule la normalisation est appliquée.

Le mode `class_mode="categorical"` prépare les étiquettes pour une classification à plusieurs classes.

### Étape 4 — Construire le CNN

Le modèle `computer_vision_model` est un réseau séquentiel composé de :

1. Une entrée de forme `(224, 224, 3)` pour une image couleur.
2. Des couches `Conv2D` qui apprennent des caractéristiques visuelles.
3. Des activations `ReLU` qui introduisent de la non-linéarité.
4. Des couches `MaxPool2D` qui réduisent la taille des cartes de caractéristiques.
5. Une couche `Flatten` qui transforme les cartes en vecteur.
6. Une couche `Dense(10, activation="softmax")` qui produit un score de probabilité pour chacune des dix classes.

### Étape 5 — Compiler le modèle

Le modèle utilise :

- **Fonction de perte** : `categorical_crossentropy`, adaptée à une classification multiclasse avec des étiquettes catégorielles.
- **Optimiseur** : `Adam`.
- **Métrique** : `accuracy`.

### Étape 6 — Entraîner le modèle

L’appel à `fit()` entraîne le réseau pendant **50 époques**, avec `train_data` pour l’entraînement et `test_data` comme données de validation dans le notebook.

> Remarque méthodologique : utiliser les données de test comme données de validation pendant le développement signifie qu’elles ne constituent plus un test final totalement indépendant. Pour une évaluation finale plus rigoureuse, il est préférable de disposer d’un ensemble de validation distinct et de conserver le test pour la fin.

### Étape 7 — Examiner les résultats

L’historique renvoyé par `fit()` est converti en DataFrame Pandas, puis affiché et tracé. Les courbes permettent d’examiner l’évolution des métriques enregistrées pendant l’entraînement, notamment la perte et l’exactitude pour l’entraînement et la validation.

Les performances réelles ne sont pas indiquées dans ce README, car elles dépendent de l’exécution du notebook.

### Étape 8 — Prédire une nouvelle image

- `load_and_prep_img()` lit l’image, la décode, la redimensionne en `224 × 224` et normalise ses pixels.
- `pred_and_plot()` ajoute une dimension de batch, appelle `model.predict()`, choisit la classe ayant le score le plus élevé et affiche l’image avec la classe prédite.

Le notebook contient un exemple d’appel sur `pizza.jfif`. Cette image doit être présente dans le répertoire courant pour que l’exemple fonctionne.

## 6. Installation et exécution

### Prérequis

Installer Python puis Jupyter Notebook, ou utiliser VS Code avec l’extension Jupyter.

Dans un terminal, installer les bibliothèques utilisées :

```bash
python -m pip install tensorflow numpy matplotlib pandas jupyter
```

### Lancer le notebook

1. Télécharger ou cloner le dépôt GitHub contenant le projet.
2. Ouvrir le dossier du projet dans VS Code.
3. Sélectionner le noyau Python dans lequel les bibliothèques ont été installées.
4. Ouvrir `Projet_Deep_Learning.ipynb`.
5. Exécuter les cellules dans l’ordre, du début à la fin.

Le notebook télécharge et extrait les données. Une connexion Internet est donc nécessaire lors de cette étape. Vérifier également que le disque dispose de suffisamment d’espace.

## 7. Arborescence indicative

```text
projet-deep-learning/
├── Projet_Deep_Learning.ipynb
├── README.md
├── 10_food_classes_all_data.zip    # téléchargé par le notebook
├── 10_food_classes_all_data/       # créé après extraction
└── pizza.jfif                      # image d’exemple pour la prédiction
```

Les deux éléments de données sont générés ou ajoutés localement ; ils ne sont pas nécessairement présents dans le dépôt.

## 8. Points à vérifier

- La cellule d’extraction utilise `wget` seulement si l’archive n’existe pas. Sous Windows, si cette commande échoue, relancer la cellule de téléchargement utilisant `urllib.request.urlretrieve`, puis relancer l’extraction.
- Vérifier que `matplotlib` et les autres dépendances sont installés dans **le même environnement Python que le noyau du notebook**.
- Le code de prédiction suppose que le modèle produit plusieurs classes et utilise les noms de classes lus depuis les dossiers d’entraînement. Il est important de conserver le même ordre de classes entre l’entraînement et la prédiction.
- La cellule de prédiction utilise `pizza.jfif` : remplacer ce chemin par celui d’une image existante si nécessaire.
- Le notebook définit l’entraînement et la prédiction, mais ne contient pas de sauvegarde explicite du modèle entraîné.

## 9. Limites actuelles

- Le notebook ne sauvegarde pas explicitement le modèle après l’entraînement.
- L’évaluation utilise `test_data` comme validation durant l’entraînement.
- Aucune matrice de confusion ni rapport détaillé par classe n’est produit dans les cellules visibles du notebook.
- Les résultats chiffrés ne peuvent être affirmés qu’après exécution du notebook.

## 10. Améliorations possibles

- Séparer les données d’entraînement, de validation et de test.
- Évaluer le modèle final avec une matrice de confusion et des métriques par classe.
- Sauvegarder le modèle entraîné avec `model.save(...)`.
- Ajouter une fonction de prédiction qui affiche également le score associé à la classe prédite.
- Comparer ce CNN construit à partir de zéro avec une approche de transfert d’apprentissage.

## 11. Auteur

Projet de Deep Learning / Computer Vision réalisé avec Python, TensorFlow et Keras.

---

**Mots-clés :** Deep Learning, Computer Vision, CNN, TensorFlow, Keras, classification d’images, food image classification.
"""

out = Path("/mnt/data/README.md")
out.write_text(content, encoding="utf-8")
print(f"README créé : {out} ({out.stat().st_size} octets)")
#   c l a s s i f i c a t i o n - i m a g e s - a l i m e n t a i r e s  
 