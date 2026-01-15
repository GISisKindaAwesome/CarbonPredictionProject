import os
import streamlit as st
import rasterio
from rasterio.warp import transform_bounds
import leafmap.foliumap as leafmap
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

st.title("Classification")
st.write("To be able to predict the amount of Carbon stored on the forests of Samos a forest classification map is needed. This can then be used for area calculation and other calculations. Copernicus provides a classification map called Corine Land Cover which could have been used. However, this map would have been less precise, as it  shows more general world coverage and originates  from 2018. No other classification maps/data exists for the island of Samos, so  for a more recent and a more accurate classification it was decided to make a new forest classification map based on Sentinel 2 imagery. Although various classification models do exist,a random forest algorithm was chosen instead, because it has high dimensionality and can thus  process the high amount of bands while  reducing noise bias. It can also capture complex boundaries better, which is particularly useful  for the forests on Samos.")
st.write("Most of the steps on this page were done with the use of the Google Earth Engine API in Python.")

st.space(size="small")

st.subheader("Gathering imagery")
st.write("""
         For the gathering of the imagery the Sentinel 2 Surface Reflectance(SR) was used with the following parameters: 
         - Start date: 01-03-2024
         - End date: 28-02-2025
         - Maximum cloud coverage: 40%
         """)

st.space(size="small")

st.subheader("Apply cloud mask")
st.write("It is almost impossible to get cloud free imagery and to still get a good temporal extent that's why it's important to apply a cloud mask. For that the following tutorial was used for the cloud mask: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless. The tutorial uses the S2 cloud probability dataset (s2cloudless), together with various thresholds to mask out clouds and their shadows. These thresholds can be changed, but for this project the default thresholds were used.")

st.space(size="small")

st.subheader("Compositing imagery")
st.write("""
         With the gathered and masked images, composites were made for each season. This was done to keep the processing time lower, since there was no need for more images for classification. As a lot of differences could be found throughout the seasons, the images were composited based separately per season, following a median composite to remove any outliers.
         The seasons ranged from:
         - Spring: March till May 
         - Summer: June till August
         - Autumn: September till November
         - Winter: December till February
         """)

st.space(size="small")

st.subheader("Indices & Tasseled Cap Transformation")
st.write("To have a more accurate classification, various indices and the tasseled cap were calculated using the seasonal composites Before being able to calculate these, the SWIR bands had to be resampled to ten meters.These indices are the following:")
st.write("""
        - Normalized Difference Vegetation Index (NDVI)
            - Measures vegetation health and density
            - Formula: NIR-Red/NIR+Red
        - Normalized Difference Water Index (NDWI)
            - Measures water content in water bodies
            - Formula: NIR-SWIR/NIR+SWIR
        - Normalized Difference Built-up Index (NDBI)
            - Measures built-up areas
            - Formula: SWIR-NIR/SWIR+NIR
        - Enhance Vegetation Index (EVI)
            - Measures vegetation health and density but better handles soil reflection, atmospheric effects and dense vegetation
            - Formula: 2.5*(NIR-Red/NIR+6*Red-7.5*Blue+1)
        - Soil Adjusted Vegetation Index (SAVI)
            - Measures vegetation greenness while correcting for the brightness of the underlying soil
            - Formula: (NIR-Red)*(1+L)/NIR+Red+L
                - L is the soil adjustment factor where 0.5 is the default and can be changed per area of interest
        - Tasseled Cap Transformation
            - This is a spectral analysis technique which converts spectral data into three components which signify differen properties.
            - Every satellite has different coefficients which need to be used to apply the transformation correctly. For this project the following research was used to get the coefficients for Sentinel 2:
                - https://ieeexplore.ieee.org/document/8836649
            - The three components are as following:
                - To calculate the components, the values first need to be converted to true reflectance values. This was done by multiplying it by 0.0001
                - Greenness
                    - Measures vegetation health and density
                    - Formula: -0.2848*Blue-0.2435*Green-0.5436*Red+0.7243*NIR+0.0840*SWIR1-0.1800*SWIR2
                - Wetness
                    - Measures soil moisture or water content
                    - Formula: 0.1509*Blue+0.1973*Green+0.3279*Red+0.3406*NIR-0.7112*SWIR1-0.4572*SWIR2
                - Brightness 
                    - Measures soil or surface brightness
                    - Formula: 0.3037*Blue+0.2793*Green+0.4743*Red+0.5585*NIR+0.5082*SWIR1+0.1863*SWIR2
         """)

st.space(size="small")

st.subheader("Training samples")
st.write("Since the classification model is a supervised model, features are needed to distinguish between classes, so that the model has something to base its prediction on. Based on imagery the following features were chosen:")
training_samples = leafmap.Map(
    zoom_control=True, 
    attribution_control=False,   
    draw_control=False,          
    measure_control=False,       
    locate_control=False,        
    scale_control=False    
)
training_samples.add_basemap("SATELLITE")
training_samples_file = "Data/TrainingSamples/TrainingSamples.shp"
sample_style = {
    "stroke": True,
    "color": "#ff0000",
    "weight": 2,
    "opacity": 1,
    "fill": True,
    "fillColor": "#ff0000",
    "fillOpacity": 0.5,
}
training_samples.add_shp(training_samples_file, 
          layer_name="Training samples",
          style = sample_style)  
training_samples.to_streamlit()
st.write("These features were then  split into a random 70:30 training split, where 70% of the data was used for the training of the random forest model and 30% was  used for the testing of the model.")

st.space(size="small")

st.subheader("Vegetation mask")
st.write("In order to only classify the forests, a mask needed to be applied. This was done by using various summer indices with thresholds, so that only the forests would remain as the remaining pixels. This vegetation mask was set up as following:")
st.write("""
         Everything that fell under these thresholds was masked out:
         - NDVI: <0.70
         - NDBI: >0
         - NDWI: >-0.3
         - Greenness: <0.10
         - Brightness: <0.20 or >0.50
        """)

st.space(size="small")

st.subheader("Accuracy score")
st.write("After running the classification on the training samples, different accuracy tests were applied to the training and test samples.")
st.write("The proportion of classifications (total accuracy) that were correct was 0.9927, or99.27%")
st.write("The table below (confusion matrix) shows the classification model's predicted results against the actual outcomes:")
con_matrix = [[0, 0, 0],
               [0, 1413, 3],
               [0, 12, 627]]
con_df = pd.DataFrame(con_matrix, 
                       columns=["Predicted 0", "Predicted 1", "Predicted 2"],
                       index=["Actual 0", "Actual 1", "Actual 2"])
con_df = con_df.loc[["Actual 1", "Actual 2"], ["Predicted 1", "Predicted 2"]]
con_df = con_df.rename(index={"Actual 1": "Actual Coniferous forest", "Actual 2": "Actual Broadleaf forest"},
                         columns={"Predicted 1": "Predicted Coniferous forest", "Predicted 2": "Predicted Broadleaf forest"})
st.dataframe(con_df)

st.space(size="small")
         
st.subheader("Final classification map")
st.write("The random forest classification led to this map as a final result:")

#---Interactive map but raster doesn't show up----
#classification = leafmap.Map(
#    zoom_control=True, 
#    attribution_control=False,   
#    draw_control=False,          
#    measure_control=False,       
#    locate_control=False,        
#    scale_control=False  
#)
#classification.add_basemap("SATELLITE")
#clas_file = "Data/Forest_classification.tif"
#classification.add_raster(
#            clas_file,
#            layer_name="Forest classification",
#           palette=["#00000000", "#006400", "#90ee90"],
#           nodata=0
#)

#legend_dict = {
#    "Coniferous forest": "#006400",
#    "Broadleaf forest": "#90ee90"
#}
#classification.add_legend(
#    title="Forest types",
#    legend_dict=legend_dict
#)

#classification.to_streamlit()

clas_path = "Data/Forest_classification.tif"
samos_path = "Data/SamosIsland/SamosGreekGrid.shp"
gdf = gpd.read_file(samos_path)
gdf_wgs84 = gdf.to_crs(epsg=4326)
with rasterio.open(clas_path) as src:
    band = src.read(1)                  
    raster_crs = src.crs
    extent = transform_bounds(raster_crs, "EPSG:4326", *src.bounds)
         
masked = np.ma.masked_equal(band, 0)
colors = [
    (0/255.0, 100/255.0, 0/255.0),
    (144/255.0, 238/255.0, 144/255.0) 
]
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(8, 6))

fig.patch.set_facecolor("#add8e6")
ax.set_facecolor("#add8e6") 
gdf_wgs84.plot(ax=ax, facecolor="lightgrey", edgecolor="black", linewidth=1, zorder=1)
left, bottom, right, top = extent[0], extent[1], extent[2], extent[3]
ax.imshow(
    masked,
    cmap=cmap,
    vmin=1,
    vmax=2,
    extent=(left, right, bottom, top),
    origin="upper",
    interpolation="nearest",
    zorder=2
)

ax.set_xlim(left, right)
ax.set_ylim(bottom, top)

ax.axis("off")

legend_items = [
    Patch(facecolor=colors[0], edgecolor="k", label="Coniferous forest"),
    Patch(facecolor=colors[1], edgecolor="k", label="Broadleaf forest"),
    Patch(facecolor="lightgrey", edgecolor="black", label="Samos island")
]
legend = ax.legend(
    handles=legend_items,
    loc="lower right",
    title="Legend",
    fontsize=9,           
    frameon=True,
    framealpha=0.9,
    handlelength=1, 
    handleheight=1,
    borderpad=0.4,
    labelspacing=0.3
)

from matplotlib.patches import Polygon

def add_north_arrow(ax, location=(0.08, 0.90), size=0.05):
    """
    Adds a cartographic north arrow to the axes.
    location: (x, y) in axis fraction coordinates
    size: arrow size relative to axes
    """
    x, y = location
    w = size / 2
    h = size

    arrow = Polygon(
        [
            (x, y),
            (x - w, y - h),
            (x, y - h * 0.75),
            (x + w, y - h)
        ],
        closed=True,
        facecolor="black",
        edgecolor="black",
        transform=ax.transAxes,
        zorder=6
    )

    ax.add_patch(arrow)

    ax.text(
        x, y + size * 0.15,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        zorder=6
    )
add_north_arrow(ax)

st.pyplot(fig)

st.space(size="small")

st.subheader("Click below to view the final calculations to see the prediction of the total amount of carbon stored")
st.page_link(
    "pages/5_Total_carbon_stored.py",
    label="-> Carbon prediction"
)













































