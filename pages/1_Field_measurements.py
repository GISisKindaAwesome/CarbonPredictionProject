import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd

st.title("Field measurements")

st.write("To gather data on the forests of Samos, some field measurements were needed. I selected the Nightingale Valley as the location of data selection, as this valley was closely located to the Archipelagos research base while also containing both broadleaf and coniferous forests. ")


st.header("Methodology")
st.write("The following methodology was used to measure the trees:")
st.write("For each forest type six plots were selected within these plots, the height and the circumference  for each tree was measured in meters. The height was calculated by using the GLOBE observer app, while the circumference got measured using a measuring tape. For the height of the trees, the app could give some skewed results, if this was the case an estimation was made instead, based on previous height measurements. The two forest types had their own plot sizes and minimum measurement heights:")
st.write("- For coniferous forest a plot size of 20x20 meters was used where trees needed to have a minimum circumference at breast height of 31 cm to be measured.")
st.write("- For broadleaf forest a plot size of 10x10 meters was used where the trees needed to have a minimum circumference at breast height of 31 cm to be measured.")
st.write("Due to time restrictions and higher density in broadleaf forests, a different plot size was chosen.")

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

