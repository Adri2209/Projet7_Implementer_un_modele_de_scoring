🏦 Projet 7 – Implémentez un modèle de scoring
📘 Contexte du projet

"Prêt à dépenser" est une société financière qui propose des crédits à la consommation à des clients ayant peu ou pas d’historique de prêt.
L’objectif est de mettre en place un outil de scoring crédit capable de :

prédire la probabilité de remboursement d’un crédit ;

expliquer les décisions de manière transparente aux chargés de clientèle et aux clients.

🎯 Objectifs

Construire un modèle de scoring prédictif de la probabilité de défaut d’un client.

Analyser les features importantes :

globalement (importance globale SHAP)

localement (importance individuelle par client)

Mettre le modèle en production via une API Flask déployée sur Microsoft Azure.

Développer un dashboard Streamlit interactif pour visualiser les résultats et expliquer les prédictions.

Mettre en œuvre une approche MLOps complète (suivi des expériences, API, dashboard, monitoring du data drift, tests unitaires).

🧠 Jeu de données

Les données proviennent du concours Kaggle
📂 Home Credit Default Risk

Elles contiennent :

des informations personnelles, financières et comportementales sur les clients,

des historiques de crédit et de remboursement,

et des agrégations issues de plusieurs tables liées (application, bureau, POS_CASH, etc.).

| **Domaine**                   | **Outils**                                     |
| ----------------------------- | ---------------------------------------------- |
| 🐍 Langage principal          | Python 3.11                                    |
| 🤖 Machine Learning           | `scikit-learn`, `lightgbm`, `imbalanced-learn` |
| 🔍 Interprétabilité           | `SHAP`, `matplotlib`, `plotly`                 |
| 🧩 API backend                | `Flask`, `gunicorn`                            |
| 📊 Dashboard frontend         | `Streamlit`                                    |
| ☁️ Environnement Cloud        | Microsoft Azure (App Service)                  |
| 📈 Suivi expérimental (MLOps) | `MLflow`                                       |
| 🧪 Tests unitaires            | `pytest`                                       |
| 💾 Gestion de version         | `Git / GitHub`                                 |

🧩 Architecture du projet
OC-P7-Implémentez-un-modèle-de-scoring/
│
├── api.py                       # API Flask - endpoints de scoring et SHAP
├── app_streamlit.py              # Dashboard Streamlit connecté à l'API
├── test_api.py                   # Tests unitaires Flask
├── test_dashboard.py             # Tests unitaires Streamlit
├── fonctions.py                  # Fonctions utilitaires du notebook
├── requirements.txt              # Dépendances du projet
├── startup.txt                   # Commandes de démarrage Azure
├── Adriana_Tint_modelisation.ipynb
├── Adriana_Tint_modelisation_suite.ipynb
└── README.md                     # Ce fichier

🚀 Déploiement

🔹 API en production (Azure)

🌐 https://implementer-un-modele-de-scoring-b6fwe6eegaamhkdh.francecentral-01.azurewebsites.net/
L’API est actuellement déployée sur Microsoft Azure App Service et accessible publiquement.

Endpoint principal disponible :
Méthode	Endpoint	Description
GET	/	Retourne le message : "API pour prédire l'accord d'un prêt" — indiquant que l’API est opérationnelle et prête à recevoir des extensions futures (prédiction, analyse SHAP, etc.).

Cette étape valide la mise en ligne correcte de l’API et le bon fonctionnement du déploiement sur le Cloud.
Les autres endpoints (comme /prediction ou /client_info/<id>) seront ajoutés ultérieurement lors de la phase suivante du projet.

🔹 Dashboard Streamlit

Permet de :

sélectionner un client,

visualiser sa probabilité de remboursement,

explorer ses informations personnelles,

afficher les graphiques SHAP globaux et locaux.

🧪 Tests unitaires

Des tests automatisés ont été implémentés avec pytest pour :

vérifier la stabilité de l’API Flask (test_api.py),

simuler les appels API depuis le dashboard Streamlit (test_dashboard.py).

✅ Tous les tests passent avec succès :

11 passed, 0 failed

📊 Exemple de résultats

(optionnel : ajoute ici une capture d’écran de ton dashboard Streamlit ou de ton SHAP plot)

Exemple :

🧭 Compétences validées

Concevoir et déployer un modèle de scoring de crédit.

Développer et exposer une API d’inférence.

Créer un dashboard interactif explicatif.

Mettre en place une approche MLOps complète.

Versionner et tester le code avec Git et pytest.

Déployer un service ML sur Azure App Service.

👤 Auteur

Adriana TINT
Projet réalisé dans le cadre de la formation Data Scientist – OpenClassrooms
📅 Date : 2025
📍 Technologies : Python, Flask, Streamlit, MLflow, Azure

