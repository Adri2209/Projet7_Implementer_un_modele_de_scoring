from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# ------------------------------
# 1️⃣ Charger les données
# ------------------------------
test_data = pd.read_csv('test_mean_sample.csv')
test_data['client_id'] = range(1, len(test_data) + 1)

# ------------------------------
# 2️⃣ Charger le pipeline entraîné (.joblib)
# ------------------------------
# Le pipeline contient déjà : StandardScaler + SMOTE + LGBMClassifier
model = joblib.load("pipeline_LGBM_prediction.joblib")

# ------------------------------
# 3️⃣ Construire SHAP avec le même modèle
# ------------------------------
# Pour SHAP, on récupère les features d'entraînement
train_data = pd.read_csv('train_mean_sample.csv')
X_train = train_data.drop(['TARGET'], axis=1)

explainer = shap.Explainer(model.named_steps['lgbmclassifier'], model.named_steps['standardscaler'].transform(X_train))
shap_values_train = explainer(model.named_steps['standardscaler'].transform(X_train), check_additivity=False)
global_shap_values = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(
    list(zip(X_train.columns, global_shap_values)),
    columns=['feature', 'importance']
).sort_values(by='importance', ascending=False)


# ------------------------------
# ROUTES
# ------------------------------
@app.route('/')
def home():
    return 'API de scoring — modèle LightGBM chargé via Joblib'


@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "ok"}, 200


@app.route('/check_client/<int:client_id>', methods=['GET'])
def check_client_id(client_id):
    return jsonify(client_id in list(test_data['client_id']))


@app.route('/client_info/<int:client_id>', methods=['GET'])
def get_client_info(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    return client_data.to_dict(orient='records')[0]


@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.get_json()
    client_id = data.get('client_id')

    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)

    # ⚠️ Le pipeline gère déjà le scaler → juste predict_proba
    prediction = model.predict_proba(info_client)[0][1]

    return jsonify({"prediction": float(prediction)})


@app.route('/local_feature_importance/<int:client_id>', methods=['GET'])
def local_feature_importance(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    shap_values = explainer(model.named_steps['standardscaler'].transform(info_client), check_additivity=False)

    local_shap_values = np.abs(shap_values.values[0])
    local_shap_importance = pd.DataFrame(
        list(zip(info_client.columns, local_shap_values)),
        columns=['feature', 'importance']
    ).sort_values(by='importance', ascending=False)

    return jsonify(local_shap_importance.set_index('feature')['importance'].to_dict())


@app.route('/global_feature_importance', methods=['GET'])
def global_feature_importance_route():
    top = global_shap_importance.head(10)
    return jsonify(top.set_index('feature')['importance'].to_dict())


@app.route('/shap_summary_plot/<int:client_id>', methods=['GET'])
def shap_summary_plot(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    shap_values = explainer(model.named_steps['standardscaler'].transform(info_client), check_additivity=False)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values.values, info_client, plot_type="bar", max_display=10, show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({"shap_summary_plot": img_base64})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
