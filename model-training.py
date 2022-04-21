import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression  # pip install scikit-learn
from sklearn.model_selection import train_test_split
import pickle

# load data


def load_iris():
    return px.data.iris()


df_iris = load_iris()

X = df_iris.drop(columns=["species", "species_id"])
y = df_iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=70)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("works")

# save the model
with open("saved-iris-model.pkl", "wb") as file:
    pickle.dump(lr, file)
