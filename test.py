import pytest
from app import app, model_pred
import statsmodels.api as sm  # Ajoutez cette ligne

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    #assert b"Welcome" in response.data  # Assurez-vous que votre page d'accueil contient ce texte

def test_predict_route(client):
    response = client.post('/predict', data={
        'credit_lines_outstanding': 5,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 3500,
        'income': 60000,
        'years_employed': 2,
        'fico_score': 132
    })
    assert response.status_code == 200
    assert b"credit granted" in response.data or b"credit is therefore not granted" in response.data

def test_model_pred():
    constant = 133.9888
    new_data = {
        'credit_lines_outstanding': 5,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 3500,
        'income': 60000,
        'years_employed': 2,
        'fico_score': 132,
    }
    features = [constant] + list(new_data.values())
    prediction = model_pred(features)
    assert prediction == 1, "incorrect prediction"
