import pandas as pd

df = pd.read_csv("../data/creditcard.csv")

# 1. Affiche la forme du dataset
print("Shape:", df.shape)

# 2. Affiche les 5 premières lignes
print(df.head())

# 3. Vérifie les valeurs manquantes
print("Valeurs manquantes:", df.isnull().sum().sum())

# 4. Distribution de la cible (Class)
print(df["Class"].value_counts())

# 5. Proportion de fraude
print("Pourcentage de fraude:", df["Class"].mean() * 100, "%")
print(df[["Time", "Amount"]].describe())
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Fraudes dans y_train:", y_train.sum(), "/", len(y_train))
print("Fraudes dans y_test:", y_test.sum(), "/", len(y_test))
#normalisation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print(X_train[["Time", "Amount"]].describe())
# entrainementt du modele 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score

# Modèle 1 : régression logistique (simple, sensible à l'échelle, d'où la normalisation)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_score_lr = lr.predict_proba(X_test)[:, 1]

print("=== Régression logistique ===")
print(classification_report(y_test, y_pred_lr, digits=3))
print("PR-AUC:", average_precision_score(y_test, y_score_lr))

# Modèle 2 : Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_score_rf = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, digits=3))
print("PR-AUC:", average_precision_score(y_test, y_score_rf))
df = pd.read_csv("../data/creditcard.csv")  # depuis src/, ou "data/creditcard.csv" depuis la racine
fraud_example = df[df["Class"] == 1].iloc[0]
print(fraud_example.to_json())