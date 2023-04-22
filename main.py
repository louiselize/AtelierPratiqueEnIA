import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from PIL import Image
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuration de Streamlit
st.set_page_config(page_title="MonnAI", page_icon=":chart_with_upwards_trend:", layout="wide")
import streamlit.components.v1 as components

def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
# Ajout du CSS inline
st.markdown("""
<style>
    body {
        background-color: #f4f4f4;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données financières
def get_financial_data(currency_pair, start_date, end_date):
    data = yf.download(currency_pair, start=start_date, end=end_date)
    return data

# Fonction pour créer un graphique interactif
def plot_financial_data(data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]))
    fig.update_layout(title="Graphique EUR/CAD", xaxis_title="Date", yaxis_title="Valeur")
    return fig


def simple_ai_algorithm(data):
    #Si le cours de clôture est supérieur à la moyenne mobile sur 30 jours, conseil = Investir, sinon = Ne pas investir
    data["SMA30"] = data["Close"].rolling(window=30).mean()
    if data.iloc[-1]["Close"] > data.iloc[-1]["SMA30"]:
        return "Investir (vert)"
    else:
        return "Ne pas investir (rouge)"

# Fonction pour entraîner un modèle de régression linéaire et effectuer des prédictions
def predict_lin_reg(data, prediction_days):
    data["Prediction"] = data["Close"].shift(-prediction_days)
    X = np.array(data.drop(["Prediction"], 1))[:-prediction_days]
    y = np.array(data["Prediction"])[:-prediction_days]

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    X_pred = np.array(data.drop(["Prediction"], 1))[-prediction_days:]
    predictions = lin_reg.predict(X_pred)

    return predictions

def check_credentials(username, password):
    if username == "user" and password == "password":
        return True
    return False

def contact_section():
    st.header("Contact")
    st.write("N'hésitez pas à nous contacter pour toute question ou commentaire :")
    st.write("Email : contact@monnai.com")
    st.write("Téléphone : +33 1 23 45 67 89")

def team_section():
    st.header("Notre Équipe")
    st.write("Voici les membres de notre équipe talentueuse :")
    st.write("1. Kyllian BIZOT - Spécialiste en modélisation prédictive")
    st.write("2. Louise LIZE - Analyste de données et soutiens technique")
    st.write("3. Tristan GUICHARD - Expert en développement web et expérience utilisateur")


# Affichage de la page principale
def main():
    is_logged_in = st.session_state.get('logged_in', False)

    if not is_logged_in:
        st.title("Connexion")
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit_button = st.form_submit_button("Se connecter")

        if submit_button:
            if check_credentials(username, password):
                st.success("Connexion réussie")
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")
    else:
        # Ajout du menu burger
        burger_menu = st.sidebar.selectbox(
            "",
            ["Accueil", "Contact", "Notre Équipe"],
            format_func=lambda x: "≡" if x == "" else x
        )
        
        if burger_menu == "Contact":
            contact_section()
        elif burger_menu == "Notre Équipe":
            team_section()
        else:
            load_css("style.css")
        
        
        
            col1, col2, col3 = st.columns([1, 6, 1])
        
            with col1:
                st.write("")
        
            with col2:
        
                logo = Image.open("Monn_AI.png")
                st.image(logo, width=100)
        
                # Navigation des onglets en horizontale
                page = st.selectbox("", ["Graphiques", "Conseils", "Prédictions"])
        
                # Onglet Graphiques
                if page == "Graphiques":
                    st.header("Graphiques EUR/CAD")
        
                    start_date = st.date_input("Date de début", value=datetime.date.today() - datetime.timedelta(days=365))
                    end_date = st.date_input("Date de fin", value=datetime.date.today())
        
                    if start_date and end_date:
                        data = get_financial_data("EURCAD=X", start_date, end_date)
                        st.plotly_chart(plot_financial_data(data), use_container_width=True)
        
                # Onglet Conseils
                elif page == "Conseils":
                    st.header("Conseils d'investissement")
        
                    # Générer des conseils basés sur l'exemple simple d'algorithme d'IA
                    data = get_financial_data("EURCAD=X", datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())
                    advice = simple_ai_algorithm(data)
        
                    if advice == "Investir":
                        st.success("🟢 Conseil d'aujourd'hui: Investir")
                        st.info("L'algorithme prédit une hausse du cours du EUR/CAD.")
                    else:
                        st.error("🔴 Conseil d'aujourd'hui: Ne pas investir")
                        st.info("L'algorithme prédit une baisse du cours du dollar canadien.")
                # Onglet Prédictions
                elif page == "Prédictions":
                    st.header("Prédictions du cours EURCAD/EUR")
                
                    prediction_start_date = st.date_input("Date de début des prédictions", value=datetime.date.today())
                    prediction_end_date = st.date_input("Date de fin des prédictions", value=datetime.date.today() + datetime.timedelta(days=30))
                
                    data = get_financial_data("EURCAD=X", datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())
                    prediction_days = (prediction_end_date - prediction_start_date).days
                    predictions = predict_lin_reg(data, prediction_days)
                
                    # Afficher les prédictions sous la forme de rectangles stylisés
                    st.write("Calendrier des prédictions à venir:")
                
                    # Ajout de style pour les rectangles
                    rectangle_style = """
                    <style>
                        .prediction-card {
                            background-color: #f0f0f0;
                            border-radius: 10px;
                            padding: 20px;
                            margin: 10px 0;
                            width: 100%;
                            text-align: center;
                            box-shadow: 0 4px 6px 0 rgba(0, 0, 0, 0.2), 0 6px 10px 0 rgba(0, 0, 0, 0.19);
                        }
                        .prediction-date {
                            font-size: 1.2em;
                            font-weight: bold;
                        }
                        .prediction-value {
                            font-size: 1.1em;
                        }
                    </style>
                    """
                
                    st.markdown(rectangle_style, unsafe_allow_html=True)
                
                    # Afficher les rectangles avec les prédictions
                    for i, pred in enumerate(predictions):
                        date = prediction_start_date + datetime.timedelta(days=i)
                        date_str = date.strftime("%Y-%m-%d")
                        prediction_str = f"Le cours devrait être autour de : {pred:.4f}"
                        
                        card_html = f"""
                        <div class="prediction-card">
                            <div class="prediction-date">{date_str}</div>
                            <div class="prediction-value">{prediction_str}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
    
        
            with col3:
                st.write("")

if __name__ == "__main__":
    main()

