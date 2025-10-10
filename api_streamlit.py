# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt

# Charger les données
train_data = pd.read_csv('train_mean_sample.csv')
test_data = pd.read_csv('test_mean_sample.csv')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Séparer les features et target pour entraînement
X_train = train_data.drop(['TARGET', 'client_id'], axis=1)
y_train = train_data['TARGET']

# Appliquer scaler et SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Entraîner le modèle
model = LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=31, force_col_wise=True)
model.fit(X_train_resampled, y_train_resampled)

# Créer un explainer SHAP
explainer = shap.Explainer(model, X_train_scaled)
shap_values_train = explainer(X_train_scaled, check_additivity=False)
global_shap_values = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(
    list(zip(X_train.columns, global_shap_values)), 
    columns=['feature', 'importance']
).sort_values(by='importance', ascending=False)

# --- Streamlit Interface ---
st.title("Application de Prédiction de Prêt Client")

# Sélection du client
client_id = st.number_input("Entrez le client_id", min_value=1, step=1)

# Affichage des informations du client
if st.button("Afficher les informations du client"):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        st.error("Client non trouvé")
    else:
        st.write(client_data)

# Prédiction pour le client
if st.button("Prédiction du client"):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        st.error("Client non trouvé")
    else:
        X_client = client_data.drop('client_id', axis=1)
        X_client_scaled = scaler.transform(X_client)
        prediction = model.predict_proba(X_client_scaled)[0][1]
        st.write(f"Probabilité d'accord du prêt: {prediction:.2f}")

# --- Placeholders pour les charts ---
global_placeholder = st.empty()
local_placeholder = st.empty()

# Feature importance globale
if st.checkbox("Afficher l'importance globale des features (SHAP)", key="global_importance_chk"):
    if not global_shap_importance.empty:
        with global_placeholder.container():
            st.subheader("Importance globale des features")
            st.bar_chart(global_shap_importance.set_index('feature')['importance'].head(10))
    else:
        st.warning("Aucune donnée globale disponible")

# Feature importance locale
if st.checkbox("Afficher l'importance locale des features pour le client", key="local_importance_chk"):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        local_placeholder.error("Client non trouvé")
    else:
        X_client = client_data.drop('client_id', axis=1)
        X_client_scaled = scaler.transform(X_client)
        shap_values = explainer(X_client_scaled, check_additivity=False)
        local_shap_values = np.abs(shap_values.values[0])
        local_shap_importance = pd.DataFrame(
            list(zip(X_client.columns, local_shap_values)),
            columns=['feature', 'importance']
        ).sort_values(by='importance', ascending=False)
        with local_placeholder.container():
            st.subheader(f"Importance locale des features - Client {client_id}")
            st.bar_chart(local_shap_importance.set_index('feature')['importance'].head(10))
