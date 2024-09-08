from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("reglogisticmodel.pkl", "rb"))  # Modèle chargé depuis le fichier .pkl


def model_pred(features):
    """ Fonction pour prédire le résultat basé sur les features données """
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def home():
    """ Fonction pour gérer la page d'accueil """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """ Fonction pour gérer les prédictions basées sur les données soumises """
    if request.method == "POST":
        try:
            # Extraction des valeurs du formulaire
            credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
            loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
            total_debt_outstanding = float(request.form["total_debt_outstanding"])
            income = float(request.form["income"])
            years_employed = int(request.form["years_employed"])
            fico_score = int(request.form["fico_score"])

            # Constante ajoutée aux prédictions
            constant = 133.9888

            # Prédiction basée sur le modèle
            prediction = model.predict(
                [[constant, credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
            )

            # Rendu du template avec le résultat
            if prediction[0] == 1:
                return render_template(
                    "index.html",
                    prediction_text="Vous faites défaut, le crédit ne sera pas accordé."
                )
            else:
                return render_template(
                    "index.html", prediction_text="Tout va bien, crédit accordé ! :)"
                )

        except Exception as e:
            return render_template("index.html", prediction_text=f"Erreur dans les données: {e}")

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
