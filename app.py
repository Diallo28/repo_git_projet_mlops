from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("reglogisticmodel.pkl", "rb"))  # Chargement du modèle depuis le fichier pickle


def model_pred(features):
    """Fonction pour prédire le résultat basé sur les features fournies"""
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def home():
    """Route pour afficher la page d'accueil"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Route pour gérer les prédictions basées sur les données soumises via un formulaire"""
    if request.method == "POST":
        try:
            # Extraction des données du formulaire
            credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
            loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
            total_debt_outstanding = float(request.form["total_debt_outstanding"])
            income = float(request.form["income"])
            years_employed = int(request.form["years_employed"])
            fico_score = int(request.form["fico_score"])

            # Constante pour la prédiction (vous pouvez adapter cette valeur si nécessaire)
            constant = 133.9888

            # Prédiction du modèle
            prediction = model.predict(
                [[constant, credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
            )

            # Rendre le template en fonction du résultat de la prédiction
            if prediction[0] == 1:
                return render_template(
                    "index.html",
                    prediction_text="Vous faites défaut, le crédit ne sera pas accordé."
                )
            else:
                return render_template(
                    "index.html", prediction_text="Tout va bien, crédit accordé ! :)"
                )

        except ValueError as e:
            # Gestion des erreurs liées aux types de valeurs (par exemple, si on soumet des lettres à la place de chiffres)
            return render_template("index.html", prediction_text=f"Erreur dans les valeurs numériques : {e}")
        
        except KeyError as e:
            # Gestion des erreurs liées à des clés manquantes dans le formulaire
            return render_template("index.html", prediction_text=f"Clé manquante dans le formulaire : {e}")

        except Exception as e:
            # Gestion des erreurs non anticipées
            return render_template("index.html", prediction_text=f"Une erreur inattendue est survenue : {e}")

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
