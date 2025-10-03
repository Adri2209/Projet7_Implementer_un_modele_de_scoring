import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

# URL de ton API Flask
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

st.title("📊 Dashboard de Scoring Crédit")

# --- Sélection du client ---
st.sidebar.header("🔎 Sélection du client")

# Exemple de quelques IDs disponibles (tu peux les mettre à jour avec check_clients.py)
client_list = list(range(1, 21))

client_choice = st.sidebar.selectbox(
    "Choisir un client ID",
    options=client_list,
    index=0,
    key="client_selectbox"
)

client_id = st.sidebar.number_input(
    "Ou entrer un ID manuellement",
    min_value=1,
    step=1,
    key="client_id_input"
)

# --- Bouton pour charger les infos ---
if st.button("Charger les infos client", key="load_client_btn"):
    # On choisit d'abord l'ID de la selectbox si différent
    selected_id = client_choice if client_id == 1 else client_id

    # Récupérer les infos client depuis l'API
    response = requests.get(f"{API_URL}/client_info/{selected_id}")
    if response.status_code == 200:
        client_info = response.json()
        st.subheader("📌 Informations client")
        st.json(client_info)

        # Prédiction
        pred_response = requests.post(f"{API_URL}/prediction", json={"client_id": selected_id})
        if pred_response.status_code == 200:
            prediction = pred_response.json()["prediction"]
            st.subheader("🎯 Probabilité de défaut de paiement")
            st.metric(label="Score de défaut", value=f"{prediction:.2%}")

        # Local Feature Importance
        shap_local = requests.get(f"{API_URL}/local_feature_importance/{selected_id}")
        if shap_local.status_code == 200:
            st.subheader("🔎 Importance locale des features (SHAP)")
            shap_data = shap_local.json()
            shap_df = pd.DataFrame(list(shap_data.items()), columns=["Feature", "Importance"])
            st.bar_chart(shap_df.set_index("Feature"))

        # SHAP Summary Plot (image encodée en base64)
        shap_plot = requests.get(f"{API_URL}/shap_summary_plot/{selected_id}")
        if shap_plot.status_code == 200:
            st.subheader("📈 Graphique SHAP")
            img_base64 = shap_plot.json()["shap_summary_plot"]
            image = base64.b64decode(img_base64)
            st.image(io.BytesIO(image))

    else:
        st.error("❌ Client introuvable dans la base de test")

# --- Global Feature Importance ---
if st.checkbox("Afficher l’importance globale des features", key="global_importance_chk"):
    global_importance = requests.get(f"{API_URL}/global_feature_importance")
    if global_importance.status_code == 200:
        st.subheader("🌍 Importance globale des features")
        global_data = global_importance.json()
        global_df = pd.DataFrame(list(global_data.items()), columns=["Feature", "Importance"])
        st.bar_chart(global_df.set_index("Feature"))
