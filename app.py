
import streamlit as st

st.title("Prédicteur de Panier 🏀")

team_a = st.text_input("Nom de l'équipe A")
team_b = st.text_input("Nom de l'équipe B")

if st.button("Prédire"):
    st.write(f"Prédiction : {team_a} vs {team_b} -> Victoire probable de {team_a if len(team_a) > len(team_b) else team_b}")
