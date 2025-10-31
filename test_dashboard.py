import pytest
import requests
from types import SimpleNamespace

# On importe la fonction à tester depuis ton fichier Streamlit
from app_streamlit import get_api_json

# --- Test de la fonction get_api_json ---
def test_get_api_json(monkeypatch):

    # Cas 1️⃣ : Simulation d'une requête POST réussie
    def mock_post_success(url, json):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"prediction": 0.85},
            raise_for_status=lambda: None
        )

    # On remplace temporairement requests.post par notre version simulée
    monkeypatch.setattr(requests, "post", mock_post_success)

    # On appelle la fonction testée
    response = get_api_json("prediction", method="POST", payload={"client_id": 123})

    # Vérifie que la réponse est bien interprétée
    assert response == {"prediction": 0.85}

    # Cas 2️⃣ : Simulation d'une erreur HTTP (code 400)
    def mock_post_error(url, json):
        def raise_error():
            raise requests.exceptions.HTTPError("400 Error")
        return SimpleNamespace(
            status_code=400,
            json=lambda: {"error": "Bad Request"},
            raise_for_status=raise_error
        )

    # On remplace à nouveau requests.post
    monkeypatch.setattr(requests, "post", mock_post_error)

    # On appelle la fonction qui doit retourner None en cas d'erreur
    response = get_api_json("prediction", method="POST", payload={"client_id": 999})
    assert response is None

    # Cas 3️⃣ : Simulation d'une requête GET réussie
    def mock_get_success(url):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"status": "ok"},
            raise_for_status=lambda: None
        )

    monkeypatch.setattr(requests, "get", mock_get_success)

    response = get_api_json("health", method="GET")
    assert response == {"status": "ok"}


if __name__ == "__main__":
    pytest.main(["-v", __file__])