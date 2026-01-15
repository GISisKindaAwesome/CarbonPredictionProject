import streamlit as st
import pandas as pd

st.title("Statistics")
st.write("To be able to run accurate tree statistics,  the circumference in meters first needed to be converted to the diameter at breast height in centimeters. This was  done by dividing the circumference by π and multiplying this by 100 to convert it into centimeters.")


st.header("Summary statistics")
st.write("Some summary statistics on the measurements of the forest types.")

coniferous_data = {
    "Statistic": ["count","mean","std","min","25%","50%","75%","max","cv","skw"],
    "Diameter at breast height (cm)": [103.0, 38.784360, 11.517499, 15.915494, 29.443664, 38.515496, 47.109863, 66.845076, 0.296962, 0.265691],
    "Height (m)": [103.0, 16.393786, 4.793693, 5.040000, 12.850000, 16.500000, 18.725000, 32.000000, 0.292409, 0.443356]
}
df_coniferous = pd.DataFrame(coniferous_data)

broadleaf_data = {
    "Statistic": ["count","mean","std","min","25%","50%","75%","max","cv","skw"],
    "Diameter at breast height (cm)": [135.0, 14.786084, 3.543422, 9.867606, 12.095776, 14.323945, 16.552114, 26.419721, 0.239646, 0.942830],
    "Height (m)": [135.0, 7.894889, 2.278902, 3.000000, 6.500000, 7.500000, 9.220000, 13.000000, 0.288655, 0.587530]
}
df_broadleaf = pd.DataFrame(broadleaf_data)
col1, col2 = st.columns(spec=2, gap="small")
with col1:
    st.write("Coniferous forest")
    st.dataframe(df_coniferous, hide_index= True)
with col2:
    st.write("Broadleaf forest")
    st.dataframe(df_broadleaf, hide_index= True)


st.header("Plots")
st.write("Some plots to display the measurements.")
image_path = "Data/Statistics/"

col1, col2 = st.columns(spec=2, gap="small")
with col1:
    st.image(image_path + "DBH histograms.png")
with col2:
    st.image(image_path + "Height histograms.png")

col1, col2 = st.columns(spec=2, gap="small")
with col1:
    st.image(image_path + "DBH boxplots.png")
with col2:
    st.image(image_path + "Height boxplots.png")

col1, col2 = st.columns(spec=2, gap="small")
with col1:
    st.image(image_path + "Coniferous scatterplot.png")
with col2:
    st.image(image_path + "Broadleaf scatterplot.png")

st.header("Statistical analysis")
st.write("""
        For the statistical analysis of the tree measurements the Shapiro-Wilk test was applied first to check if the data was normally distributed. A Levene's test was then applied, in order to see if the variances differed. Based on these outcomes, a follow up test was performed to compare the groups. These are:
         - If the Shapiro-Wilk test is ***significant*** and the Levene's test is ***equal*** a ***T-test*** is used
         - If the Shapiro-Wilk test is ***significant*** and the Levene's test is ***unequal*** a ***Welch's T-test*** is used
         - If the Shapiro-Wilk test is ***not significant*** a ***Mann-Whitney U test*** is used
         - For the T-test the Cohen's d can be calculated and for the Mann-Whitney test the rank-biserial correlation can be calculated to determine the effect size (How big is the difference)
         """)

st.write("The results of the test were as following:")
st.space(size="small")

st.write("**DBH (cm)**")
st.write("""
         - Shapiro-Wilk test:
            - The dbh of coniferous forest is significantly normally distributed (p=0.156)
            - The dbh of broadleaf forest is not significantly normally distributed (p=6.809*10^6)
         """)
st.write("""
         - Levene's test:
            - Between the height of the two forests the variances are significantly unequal (F=115.169, p=3.855*10^22)
         """)
st.write("""
         - Since the data is not normally distributed and variances are unequal a Mann-Whitney U test was used and the Rank-biserial correlation was calculated:
            - The dbh of the forest types are separated and there is a significant difference between them (U=13757, p=3.007*10^38)
            - There is a very large effect size thus a strong difference in how the dbh of the forest types differ with coniferous forests having thicker trees(r=-0.979)
         """)

st.space(size="small")

st.write("**Height (m)**")
st.write("""
         - Shapiro-Wilk test:
            - The height of coniferous forest is not significantly normally distributed (p=0.041)
            - The height of broadleaf forest is not significantly normally distributed (p=3.340*10^⁵)
         """)
st.write("""
         - Levene's test:
            - Between the height of the two forests the variances are significantly unequal (F=31.012, p=6.960*10^8)
         """)
st.write("""
         - Since the data is not normally distributed and variances are unequal a Mann-Whitney U test was used and the Rank-biserial correlation was calculated:
            - The height of the forest types are separated and there is a significant difference between them (U=13185, p=2.123*10^32)
            - There is a very large effect size thus a strong difference in how the height of the forest types differ with coniferous forests having higher trees (r=-0.896)
         """)


st.space(size="small")


st.subheader("Click below to view carbon calculations")
st.page_link(
    "pages/3_Carbon_calculations.py",
    label="-> Carbon calculations"

)
