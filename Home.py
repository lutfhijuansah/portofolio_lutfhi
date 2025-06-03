# pages/1_Tentang_Saya.py
import streamlit as st

st.set_page_config(page_title="Tentang Saya", layout="centered")

st.title("Welcome to My Portofolio!")
st.write("Hello! I am a professional with 4 years of experience in the food logistic industry, now transitioning into roles as a Data Scientist and Data Analyst.")
st.write("Leveraging my background in food logistics, I have honed my analytical, problem-solving, and meticulous data processing skills. Currently, I am focused on integrating my industry experience with data science expertise to create data-driven solutions.")

st.subheader("My key skills include:")
st.markdown("""
- Data Analysis & Visualization
- Machine Learning & Predictive Modeling
- Database Management (SQL)
""")

st.subheader("My Mission:")
st.write("To assist companies in making better, impactful decisions through data-driven insights.")

st.write("Thank you for visiting my portfolio!")

# (Opsional) Tambahkan gambar profil atau CV singkat jika ingin
# st.image("path/to/your_profile_picture.jpg", width=200)