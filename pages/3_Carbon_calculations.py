import streamlit as st
import pandas as pd

st.title("Carbon calculations")
st.write("To be able to calculate the amount of carbon stored in the trees various calculations needed to be applied to the measurements.")
st.space(size="small")
st.header("Formula's")
st.write("""
        The measurements that were taken were the circumference at breast height in meters which were converted earlier to diameter at breast height in centimeters and the height in meters. These two variables are needed to be able to estimate the amount of carbon stored in the trees in kg. For these various formula's were used based on the following research:
        - For most of the formulas except the above ground biomass this website was used: 
            - https://www.ecomatcher.com/how-to-calculate-co2-sequestration/
        - For the above ground biomass this reasearch paper was used. The Chave's formula was mainly made for tropical forests, but can also be used for mediterranean forests:
            - https://www.researchgate.net/publication/262197290_Improved_allometric_models_to_estimate_the_aboveground_biomass_of_tropical_trees
        - To be able to get the right data for the above ground biomass the wood density of species needed to be found for this the following database was used:
            - https://www.fs.usda.gov/nrs/pubs/gtr/gtr-nrs200-2023_appendixes/gtr_nrs200-2023_appendix11.pdf
         """)

st.space(size="small")

st.subheader("Coeffiecients")
st.write("During the field work some species could be identified based on these the mean wood density per forest type could be calculated following the database mentioned before:")
wood_density_species = pd.DataFrame({
    "Species": ["Olea europaea", "Quercus ilex", "Pinus nigra", "Pinus brutia"],
    "Forest type": ["Broadleaf", "Broadleaf", "Coniferous", "Coniferous"],
    "Wood density (gcm3)": [0.76, 0.82, 0.43, 0.43]
})
st.dataframe(wood_density_species, hide_index=True)
st.write("""
         Average wood density per forest type:
         - Broadleaf: 0.79 g/cm3
         - Coniferous: 0.43 g/cm3
         """)
st.write("If the research would be conducted and the species would be unknown in a mediterranean forest the wood density value of 0.6 g/cm3 can be used. (https://pmc.ncbi.nlm.nih.gov/articles/PMC11618071/)")

st.space(size="small")

st.subheader("The formulas are as following:")
st.write("***Chave's above ground biomass***")
st.write("- 0.0673 * (wood density * dbh^2 * height)^0.976")
st.write("***Below ground biomass***")
st.write("- Above ground biomass * 0.2")
st.write("***Total biomass***")
st.write("- Above ground biomass + Below ground biomass")
st.write("***Total dry weight***")
st.write("- Total biomass * 0.725")
st.write("***Total carbon***")
st.write("- Total dry weight * 0.5")
st.write("***Total CO2***")
st.write("- Total carbon * 3.67")
st.write("- This is not needed for this project but is calculated if it would be used in further projects")

st.space(size="small")

st.header("Results")
st.write("The result of the calculations lead to this dataframe:")

df = pd.read_csv("Data/Tree carbon.csv")
df_extracted = df[["Unique ID", "Forest type", "Above ground biomass (kg)", "Below ground biomass (kg)", "Total biomass (kg)", "Total dry weight (kg)", "Total carbon (kg)", "Total CO2 (kg)"]]
st.dataframe(df_extracted, hide_index=True)

coniferous = df_extracted[df_extracted["Forest type"] == "Coniferous"]
broadleaf = df_extracted[df_extracted["Forest type"] == "Broadleaf"]
coniferous_carbon_stats = coniferous["Total carbon (kg)"].describe()
broadleaf_carbon_stats = broadleaf["Total carbon (kg)"].describe()

st.subheader("Summary statistics on the total carbon in kg")
col1, col2 = st.columns(spec=2, gap="small")
with col1:
    st.write("Coniferous forest")
    st.dataframe(coniferous_carbon_stats)
with col2:
    st.write("Broadleaf forest")
    st.dataframe(broadleaf_carbon_stats)

st.space(size="small")

st.subheader("Click below to view the Sentinel 2 imagery forest classification")
st.page_link(
    "pages/4_Classification.py",
    label="-> Classification"

)
