import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px


# -------------------------------
# 1) Configuration de l'app
# -------------------------------
st.set_page_config(
    page_title="Détection des Faux Billets - Fadel ADAM", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">💰 Détection des Faux Billets</h1>
    <p style="color: #e8f4fd; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
        Développé par <strong>Fadel ADAM</strong> | Système ML pour l'authentification des billets
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Informations du Projet")
    st.info("""
**Développeur:** Fadel ADAM

**Technologies utilisées:**
- FastAPI (Backend)
- Streamlit (Frontend)
- Machine Learning
- Plotly (Visualisations)
""")
    
    st.markdown("### 📋 Format des données")
    st.code("""
Colonnes requises:
• diagonal
• height_left
• height_right
• margin_low
• margin_up
• length
""")
    
    st.markdown("### ⚙️ Configuration API")
    # Configuration de l'URL API
try:
    if st.secrets.get("production", False):
        DEFAULT_API_URL = st.secrets["API_URL"]
    else:
        DEFAULT_API_URL = "http://127.0.0.1:8000"
except:
    DEFAULT_API_URL = "http://127.0.0.1:8000"

    api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL, label_visibility="collapsed")

    # Test de connexion à l'API
    if st.button("🧪 Tester la connexion API", use_container_width=True):
        try:
             # Timeout augmenté à 30 secondes
            response = requests.get(f"{api_url}/health", timeout=30)
            if response.status_code == 200:
                st.success("✅ Connexion API réussie!")
                health_data = response.json()
                st.json(health_data)
            else:
                st.error(f"❌ Erreur API: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Impossible de se connecter à l'API: {e}")


# Instructions
st.markdown("""
### 📤 Instructions d'utilisation
1. Préparez votre fichier CSV avec les caractéristiques géométriques des billets
2. Uploadez le fichier via le bouton ci-dessous
3. Analysez les résultats avec graphiques et statistiques
---
""")

# -------------------------------
# Upload du fichier CSV
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Choisir un fichier CSV", 
        type=["csv"],
        help="Le fichier doit contenir les colonnes: diagonal, height_left, height_right, margin_low, margin_up, length"
    )

with col2:
    if uploaded_file is not None:
        st.success("✅ Fichier chargé avec succès!")
        file_details = {"Nom du fichier": uploaded_file.name, "Taille": f"{uploaded_file.size} bytes"}
        st.json(file_details)

if uploaded_file is not None:
    try:
        # Lecture CSV auto-détection séparateur
        df = pd.read_csv(uploaded_file, sep=None, engine='python')

        # Aperçu des données
        st.markdown("### 👀 Aperçu des données")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Nombre de lignes", len(df))
        with col2: st.metric("Nombre de colonnes", len(df.columns))
        with col3: st.metric("Valeurs manquantes", df.isnull().sum().sum())
        with col4: st.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum()} bytes")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        # Vérification colonnes
        expected_cols = ["diagonal","height_left","height_right","margin_low","margin_up","length"]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Colonnes manquantes : {missing_cols}")
            st.stop()
        else:
            st.success("✅ Toutes les colonnes requises sont présentes!")

        # Statistiques descriptives
        with st.expander("📈 Statistiques descriptives"):
            st.dataframe(df.describe(), use_container_width=True)

        # Bouton prédiction
        st.markdown("### 🚀 Lancement des prédictions")
        if st.button("🔍 Analyser les billets"):
            with st.spinner("🔄 Analyse en cours..."):
                # Envoyer fichier à l'API - 
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                    response = requests.post(f"{api_url}/predict-file", files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()

                    # Résultats
                    st.success("✅ Analyse terminée!")
                    pred_df = pd.DataFrame(result["predictions"])

                    # Couleurs conditionnelles
                    def color_predictions(val):
                        return 'background-color: #d4edda; color: #155724' if val=="vrai" else 'background-color: #f8d7da; color: #721c24'
                    
                    # Formatage des résultats
                    display_df = pred_df.copy()
                    display_df['label'] = display_df['label'].str.capitalize()
                    st.dataframe(display_df.style.applymap(color_predictions, subset=['label']), 
                                use_container_width=True, height=400)

                    # Statistiques - pour utiliser les bonnes clés
                    stats = result.get("stats", {})
                    total = stats.get('total', 1)
                    n_true = stats.get('n_true', 0)
                    n_false = stats.get('n_false', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("📊 Total analysé", total)
                    with col2: st.metric("✅ Billets authentiques", n_true)
                    with col3: st.metric("❌ Faux billets", n_false)
                    with col4: st.metric("⚠️ % de faux", f"{(n_false/total)*100:.1f}%" if total > 0 else "0%")

                    # Visualisations
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_bar = px.bar(
                            x=['Authentiques', 'Faux'],
                            y=[n_true, n_false],
                            color=['Authentiques', 'Faux'],
                            color_discrete_map={'Authentiques': '#28a745', 'Faux': '#dc3545'},
                            title="Répartition des billets"
                        )
                        fig_bar.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with col2:
                        if total > 0:
                            fig_pie = px.pie(
                                values=[n_true, n_false],
                                names=['Authentiques', 'Faux'],
                                color_discrete_map={'Authentiques': '#28a745', 'Faux': '#dc3545'},
                                title="Proportion des résultats"
                            )
                            fig_pie.update_layout(height=400)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("Aucune donnée à afficher")

                    # Analyse caractéristiques
                    df_with_pred = df.copy()
                    df_with_pred['Prédiction'] = pred_df['label'].str.capitalize()
                    feature_to_plot = st.selectbox("Choisir une caractéristique:", expected_cols)
                    fig_dist = px.histogram(
                        df_with_pred, x=feature_to_plot, color='Prédiction',
                        color_discrete_map={'Vrai': '#28a745', 'Faux': '#dc3545'},
                        title=f"Distribution de {feature_to_plot}",
                        marginal="box"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Télécharger résultats
                    csv_results = pred_df.to_csv(index=False, sep=';')
                    st.download_button("📥 Télécharger les prédictions (CSV)", csv_results, "predictions_billets.csv", "text/csv")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Erreur API : {e}")
                    st.info("💡 Vérifiez que l'API est démarrée sur l'URL spécifiée")
                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")

    except Exception as e:
        st.error(f"❌ Erreur lecture CSV : {e}")
        st.info("💡 Vérifiez le format de votre fichier CSV")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>💰 <strong>Système de Détection des Faux Billets</strong> | Développé par <strong> Fadel ADAM </strong></p>
    <p>🚀 Propulsé par FastAPI + Streamlit + ML</p>
</div>
""", unsafe_allow_html=True)
