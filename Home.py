import streamlit as st

st.title("Carbon prediction project")
st.write("Welcome to the Streamlit which will take you through the project I have been working during my time at Archipelagos Institute of Marine Conservation (01/09/2025-30/01/2026).")

st.space(size="small")

st.header("Who am I")
st.write("I am Thijme Jonckers, 20 years old and from the Netherlands. Currently I'm in my last year of my Bsc International Forest and Nature Management specialising in Tropical Forestry at Van Hall Larenstein Univeristy of Applied Sciences.")
st.write("For my final year at university it is required to complete a 5 month internship. I got the oppurtunity to follow this internship at Archipelagos Institute of Marine Conservation in the GIS team. This was perfect since during my study I discovered my interest for GIS and data analysis and using this to solve real-world environmental problems. This perfectly aligns with the oppurtunity I got at Archipelagos.")

st.space(size="small")

st.header("Goal")
st.write("In my time here I worked on a project to estimate how much carbon is stored on the island of Samos in Greece. This work contributes to some of the conservation goals that Archipelagos aims to achieve. Next to that this is a final project for my internship to show how I developed my skills in my time here. But also to show how I adapted my knowledge I gathered during the past three years at university. During my stay here I also needed to work on personal goals setup by the univeristy and myself to further develop myself. That is why I decided to do this project in Python since I had no previous knowledge on using the programming language and it playing a essential part in the GIS sphere. That is why I decided to learn how to adapt my GIS knowledge by use of Python.")

st.space(size="small")

st.header("The project")
st.write("The following steps were undertaken to complete this project:")

st.page_link(
    "pages/1_Field_measurements.py",
    label="-> Field measurements in the Nightingale Valley"
)

st.page_link(
    "pages/2_Statistics.py",
    label="-> Statistics run on the field measurements"
)

st.page_link(
    "pages/3_Carbon_calculations.py",
    label="-> Carbon calculation based on the field measurements"
)

st.page_link(
    "pages/4_Classification.py",
    label="-> Random forest classification of Sentinel-2 imagery"
)

st.page_link(
    "pages/5_Total_carbon_stored.py",
    label="-> Use classification map and carbon calculations to predict total carbon stored"
)

st.space(size="small")

st.subheader("Disclaimer")

st.write("The results of this project are a prediction and not a precise conclusion on the total amount of carbon stored. It is meant to give more an idea of the amount then a final conclusion.")
