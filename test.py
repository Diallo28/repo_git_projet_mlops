from app import model_pred

# Données de test pour prédiction
new_data = {
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 3500,
    'income': 60000,
    'years_employed': 2,
    'fico_score': 400,
}


def test_predict():
    """ Test de la fonction de prédiction """
    prediction = model_pred(new_data)
    assert prediction == 1, "Prédiction incorrecte"