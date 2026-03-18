import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("fraud_dataset_1000.csv")
X = df.drop("fraud", axis=1)
y = df["fraud"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained successfully ✅")
