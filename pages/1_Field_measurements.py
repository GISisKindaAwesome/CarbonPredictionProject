import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd

st.title("Field measurements")

st.write("To gather data on the forests of Samos some field measurements were needed. Close to the base where Archipelagos is located a valley can be found called Nightingale valley. The field measurements where conducted in this valley for both forest types, coniferous and broadleaf forest.")


st.header("Methodology")
st.write("To conduct the measurements a specific methodology was used.")
st.write("For each forest type 6 plots were measured where the height in meters and the circumference in meters got gathered per tree. For the circumference measuring tape was used and to calculate the height the GLOBE observer app was used. For the height the app could give some skewed results, if that was the case an estamiation was made based on previous height measurements. The two forest types had their own plot sizes and minimum measurement heights. These are as following:")
st.write("- For coniferous forest a plot size of 20x20 meters was used where trees needed to have a minimum circumference at breast height of 31 cm to be able to be measured.")
st.write("- For broadleaf forest a plot size of 10x10 meters was used where the trees needed to have a minimum circumference at breast height of 31 cm to be able to be measured.")
st.write("A different plot size was chosen for broadleaf forest due to the density of theses forests being higher and being time restricted.")

st.header("Nightingale valley and the plots")

night_file = "Data/NightingaleValley/NightingaleValley.shp"
measurement_file = "Data/FieldMeasurementLocations/FieldMeasurementLocations.shp"

study_area = leafmap.Map(
    zoom_control=True, 
    attribution_control=False,   
    draw_control=False,          
    measure_control=False,       
    locate_control=False,        
    scale_control=False    
)

study_area.add_basemap("SATELLITE")

night_style = {
    "stroke": True,
    "color": "#ff0000",
    "weight": 2,
    "opacity": 1,
    "fill": False,
}

study_area.add_shp(night_file, 
          layer_name="Nightingale Valley",
          style = night_style,
          info_mode = None)                

study_area.add_shp(measurement_file, 
          layer_name="Field Measurements",)


legend = {
    "Nightingale valley": "#e41a1c"
}

study_area.add_legend(legend_dict = legend)

study_area.to_streamlit()

st.header("Measurements")
st.write("The following is the raw data collected:")

tree_file = "Data/Tree measurements.csv"

tree_df = pd.read_csv(tree_file)

tree_display = tree_df.drop(columns=["Team", "Writer", "DBH measurer", "Height measurer"])

tree_orderd = tree_display[["Unique ID", "Circumference at breast height (m)", "Height (m)", "Forest type", "Plot", "Plot size (m)", "Plot tree ID", "X coordinate", "Y coordinate", "Notes"]]

st.dataframe(tree_orderd, hide_index= True)

st.subheader("Click below to view statistics")
st.page_link(
    "pages/2_Statistics.py",
    label="-> Project statistics"

)
