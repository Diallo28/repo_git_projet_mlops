import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import pickle

# Chemin du fichier CSV
directory = 'C:\\Users\\maumy\\OneDrive\\Bureau\\DU SDA\\Loan_Data.csv'

# Fonction pour charger les données
def load_and_display_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Titre de l'application
st.title("notre application pour le projet ML ops : modèle régression logistique")

# Afficher le répertoire actuel pour vérifier le chemin
current_directory = os.getcwd()
st.write("Répertoire actuel :", current_directory)

try:
    # Charger le fichier CSV
    data = pd.read_csv(directory)
    st.write("Données chargées avec succès.")
except FileNotFoundError:
    st.error("Fichier introuvable. Vérifiez le chemin et réessayez.")

if 'default' not in data.columns:
    st.error("La colonne 'default' n'existe pas dans les données.")
else:
    st.write("Tracer les paires de relations entre les variables avec la variable cible 'default'.")

# Créer des onglets pour naviguer
tabs = ["Aperçu des données", "Informations sur les données", "Statistiques descriptives", "Graphiques", "modèle de prédiction", "test prédiction"]
tab = st.selectbox("Sélectionnez un onglet", tabs)

if tab == "Aperçu des données":
    st.write("## Aperçu des données")
    
    # Affichage de l'aperçu des données après suppression de 'customer_id'
    st.write("Aperçu des données après suppression de 'customer_id' :")
    if 'customer_id' in data.columns:
        data = data.drop(columns=['customer_id'])  # Suppression de la colonne 'customer_id'
        st.write("La colonne 'customer_id' a été supprimée.")
    
    # Affichage des premières lignes du DataFrame
    st.write(data.head())

elif tab == "Informations sur les données":
    st.write("## Informations sur les données")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

elif tab == "Statistiques descriptives":
    st.write("## Statistiques descriptives")
    st.write(data.describe())

elif tab == "Graphiques":
    st.write("## Graphiques")

    # Vérifier que la colonne 'default' existe dans le DataFrame
    if 'default' in data.columns:
        # Créer un graphique de la distribution de la variable 'default'
        plt.figure(figsize=(6, 4))
        plt.hist(data['default'], bins=2, edgecolor='black')  # Histogramme de la colonne 'default'
        plt.title('Distribution de la variable cible')
        plt.xlabel('Défaut (1) ou pas de défaut (0)')
        plt.ylabel('Nombre de clients')
        plt.xticks([0, 1], ['Pas de défaut (0)', 'Défaut (1)'])
        
        # Afficher le graphique dans Streamlit
        st.pyplot(plt)
    else:
        st.error("La colonne 'default' n'existe pas dans les données.")

elif tab == "modèle de prédiction":
    X = data.drop('default', axis=1)  # Variables prédictives
    y = data['default']  # Variable cible

    # Ajouter une constante pour l'interception
    X = sm.add_constant(X)

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Paramètres pour le modèle de régression logistique
    parametres = {
        'method': 'newton',  # Méthode d'optimisation
        'maxiter': 100,      # Nombre maximum d'itérations
        'disp': False        # Affichage des informations d'optimisation
    }

    # Entraîner le modèle de régression logistique
    modele = sm.Logit(y_train, X_train)
    result = modele.fit(**parametres)

    # on affiche le résumé du modèle dans Streamlit
    st.title('Résultats du Modèle de Régression Logistique')
    st.subheader('Résumé du Modèle')
    st.text(result.summary())

    # Prédictions du modèle sur l'ensemble d'entraînement et de test
    y_train_pred = result.predict(X_train)
    y_test_pred = result.predict(X_test)

    # Courbes ROC et AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)

    # pour tracer les courbes ROC
    st.subheader('Courbes ROC')
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='ROC curve (train) (AUC = %0.2f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='green', lw=2, label='ROC curve (test) (AUC = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Matrice de confusion pour l'ensemble de test
    st.subheader('Matrice de Confusion (Test)')
    y_pred_test = (y_test_pred > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_test)
    st.write(cm)

    # Matrice de confusion pour l'ensemble d'entraînement
    st.subheader('Matrice de Confusion (Train)')
    y_pred_train = (y_train_pred > 0.5).astype(int)
    cm_train = confusion_matrix(y_train, y_pred_train)
    st.write(cm_train)

elif tab == "test prédiction":
    st.title("Prédiction de défaut de crédit")

    # Chargement  modèle
    model_path = r"C:\Users\maumy\Documents\streamlit_projetcs\reg_logistic_model.pkl"
    model = pickle.load(open(model_path, "rb"))

    def model_pred(features):
        test_data = pd.DataFrame([features])    
        test_data = sm.add_constant(test_data, has_constant='add')
        prediction = model.predict(test_data)
        return int(prediction[0])

    # les entrées
    credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, step=1)
    loan_amt_outstanding = st.number_input("Montant du prêt restant", min_value=0.0)
    total_debt_outstanding = st.number_input("Dette totale restante", min_value=0.0)
    income = st.number_input("Revenu", min_value=0.0)
    years_employed = st.number_input("Nombre d'années d'emploi", min_value=0, step=1)
    fico_score = st.number_input("Score FICO", min_value=300, max_value=850, step=1)

    # Bouton pour prédire
    if st.button("Prédire"):
        # Créer la liste des caractéristiques à partir des entrées de l'utilisateur
        features = [credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]

        # Faire la prédiction
        prediction = model_pred(features)

        # Afficher le résultat
        if prediction == 1:
            st.error("Vous êtes en défaut. Crédit non accordé.")
        else:
            st.success("Vous êtes bien. Crédit accordé !")

