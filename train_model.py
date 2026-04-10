import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = {
    "amount": [1000, 5000, 12000, 20000, 3000, 15000],
    "previous_claims": [0, 1, 3, 5, 0, 4],
    "is_injury": [0, 1, 1, 1, 0, 1],
    "fraud": [0, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "previous_claims", "is_injury"]]
y = df["fraud"]

model = LogisticRegression()
model.fit(X, y)

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as fraud_model.pkl")