import pandas as pd
import numpy as np
from pathlib import Path
from  sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from custom_transformers import CleanBrandTransformer


# Caminho relativo ao dataset
RAW = Path("../01_data-preprocessing/cars.csv")
df = pd.read_csv(RAW)

# Corrigir tipos numéricos
for col in ["cubicinches", "weightlbs"]:
    df[col] = df[col].replace({',': '', ' ': np.nan, '': np.nan}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].astype(float).mean())

# Features e target
target = "mpg"
num_features = ["cubicinches", "hp", "weightlbs", "time-to-60"]
cat_features = ["brand"]
pass_features = ["cylinders", "year"]

X = df[num_features + cat_features + pass_features]
y = df[target]

from sklearn.base import BaseEstimator, TransformerMixin

# Criar um pipeline de pré-processamento
preprocessor = Pipeline(steps=[
    ('clean', CleanBrandTransformer()),
    ('encode_scale', ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features),
        ('pass', 'passthrough', pass_features)
    ]))
])

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# Treino e avaliação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"R²:  {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Salvar artefatos
joblib.dump(pipe, "pipeline.pkl")
joblib.dump({
    "num_features": num_features,
    "cat_features": cat_features,
    "pass_features": pass_features,
    "target": target
}, "schema.joblib")

print("✅ Pipeline e schema salvos com sucesso!")