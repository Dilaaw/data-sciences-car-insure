import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# On importe les données du csv dans un tableau DataFrame
data = pd.read_csv("car_insurance.csv")

# On affiche la taille du tableau, ici on a 10 000 lignes pour 18 colonnes
print("Taille du tableau: ", data.shape)

# On affiche le type de chaque données
print("Type des données: ", data.dtypes)

# On affiche le nombres de valeurs manquantes par colonnes:
print("Données manquantes:\n", data.isna().sum())

# On remarque que 'credit_score' possède 982 données manquantes et que 'annual_mileage' en
# possède 957. Il faut donc les remplacer par de nouvelles valeurs numériques.

# Ici, on remplace les valeurs manquantes des valeurs numériques par la médiane
data['credit_score'].fillna(data['credit_score'].median(), inplace=True)
data['annual_mileage'].fillna(data['annual_mileage'].median(), inplace=True)

# Désormais on affiche si il y a toujours des données manquantes ou non.
print("Affichage des données manquantes après remplacement :\n", data.isna().sum())

# On affiche les histogrammes des variables numériques
#data.hist(bins=10, figsize=(20, 15))
#plt.show()

# Remplacement des valeurs aberrantes par la médiane
# On remplace les valeurs aberrantes supérieurs à 1 car cette data est un booléen, soit 0 (pas d'enfant) soit 1
# (a des enfants)
data.loc[data['children'] > 1, 'children'] = data['children'].median()
# On trouve 29 valeurs supérieurs à 13 pour la data 'speeding_violation'
data.loc[data['speeding_violations'] > 13, 'speeding_violations'] = data['speeding_violations'].median()
# 141 valeurs ont plus de 6 accidents, généralement avec autant d'accidents, une voiture est considérée comme épave.
data.loc[data['past_accidents'] > 6, 'past_accidents'] = data['past_accidents'].median()

# On retire la colonne id qui n'est pas utile pour la suite, l'id ne vas pas nous permettre de prédire quoi que ce soit
data = data.drop(['id'], axis=1)

# Nous avons des données qualitatives, nous devons donc désormais les transformer en données quantitatives
# Les colonnes à transformer sont 'vehicle_type', 'vehicle_year', 'income', 'education' et 'driving_experience'

transform_columns = ['driving_experience', 'income', 'vehicle_type', 'vehicle_year', 'education']

# On instancie le LabelEncoder pour transformer les données

label_encoder = LabelEncoder()

# On applique le LabelEncoder sur chaque colonne qualitative
for column in transform_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Désormais, on vérifie une nouvelle fois le type des données afin de voir si nous avons uniquement des données quantitatives
print("Type des données après transformation:\n", data.dtypes)

# On retire désormais la colonne outcome qui est la colonne qu'on doit prédire
# On sauvegarde dans un premier temps la colonne qu'on doit prédire
outcome = data['outcome']
# Puis on la supprime de data
data = data.drop(['outcome'], axis=1)

# Normalisation des données
scaler = StandardScaler()

# On transforme toutes nos données en float64
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print("Type des données:\n", data.dtypes)

correlations = data.corr()
print(correlations)

# On définit notre seuil de corrélation à 0.7
threshold = 0.7

# Filtrer la matrice de corrélation en fonction du seuil
high_correlations = correlations[correlations.abs() > threshold]

# Afficher la matrice de corrélation filtrée
# plt.imshow(high_correlations, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xticks(range(len(high_correlations.columns)), high_correlations.columns, rotation='vertical')
# plt.yticks(range(len(high_correlations.columns)), high_correlations.columns)
# plt.title('Matrice de corrélation avec comme seuil 0.7')
# plt.show()

# On constate que la corrélation est entre 'age' et 'driving_experience'

# Désormais, on sépare les données en 2 parties:
# Une partie test: 25% des données de test dans ce cas-ci
# Une partie entraîneemnt: 75% des données de test
X = data
y = outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# On instancie les différents modèles de régression logistique
model_logreg = LogisticRegression()
model_perceptron = Perceptron()
model_knn = KNeighborsClassifier()

# On entraîne les différents modèles
model_logreg.fit(X_train, y_train)
model_perceptron.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# On effectue une cross validation et on calcule les scores
cv = 5
scores_model_logreg = cross_val_score(model_logreg, X_train, y_train, cv=cv)
scores_model_perceptron = cross_val_score(model_perceptron, X_train, y_train, cv=cv)
scores_model_knn = cross_val_score(model_knn, X_train, y_train, cv=cv)

# On affiche les scores moyens au bout de 5 cross validation
print("Score du modèle Logistic Regression:", np.mean(scores_model_logreg, axis=0))
print("Score du modèle Perceptron:", np.mean(scores_model_perceptron, axis=0))
print("Score du modèle KNN:", np.mean(scores_model_knn, axis=0))

print("\n")

# Maintenant, on évalue les modèles sur l'ensemble des tests puis on les affiche
print("Précision de Logistic Regression:", model_logreg.score(X_test, y_test))
print("Précision de Perceptron:", model_perceptron.score(X_test, y_test))
print("Précision de KNN:", model_knn.score(X_test, y_test))

# On fait des prédictions sur le jeu de test sur chaque modèle
y_pred_logreg = model_logreg.predict(X_test)
y_pred_perceptron = model_perceptron.predict(X_test)
y_pred_knn = model_knn.predict(X_test)

print("\n")

# On calcule l'accuracy score de chaque modèle
print("Score de prédictions correctes (Logistic Regression):", accuracy_score(y_test, y_pred_logreg))
print("Score de prédictions correctes (Perceptron):", accuracy_score(y_test, y_pred_perceptron))
print("Score de prédictions correctes (KNN):", accuracy_score(y_test, y_pred_knn))

print("\n")

# On calcule la matrice de confusion de chaque modèle
print("Matrice de confusion (Logistic Regression):\n", confusion_matrix(y_test, y_pred_logreg))
print("Matrice de confusion (Perceptron):\n", confusion_matrix(y_test, y_pred_perceptron))
print("Matrice de confusion (KNN):\n", confusion_matrix(y_test, y_pred_knn))

print("\n")

# On calcule le score de précision de chaque modèle
print("Score de précision (Logistic Regression):", precision_score(y_test, y_pred_logreg))
print("Score de précision (Perceptron):", precision_score(y_test, y_pred_perceptron))
print("Score de précision (KNN):", precision_score(y_test, y_pred_knn))

print("\n")

# On calcule le recall score de chaque modèle
print("Recall score (Logistic Regression):", recall_score(y_test, y_pred_logreg))
print("Recall score (Perceptron):", recall_score(y_test, y_pred_perceptron))
print("Recal score (KNN):", recall_score(y_test, y_pred_knn))

print("\n")

# On calcule le F1 score de chaque modèle
print("F1 Score (Logistic Regression):", f1_score(y_test, y_pred_logreg))
print("F1 Score (Perceptron):", f1_score(y_test, y_pred_perceptron))
print("F1 Score (KNN):", f1_score(y_test, y_pred_knn))

