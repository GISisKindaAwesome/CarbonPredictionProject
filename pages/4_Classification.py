import streamlit as st
import leafmap.foliumap as leafmap
import pandas as pd

st.title("Classification")
st.write("To be able to predict the amount of Carbon stored on the forests of Samos a forest classification map is needed. This then can be used for area calculation and other calculations. Copernicus provides a classification map called Corine Land Cover which could be used. However, this map will be less precise due to it being more general for the world and is from 2018. No other classification maps/data exists for the island of Samos. So for more recent and a more accurate classification it was decided to make a forest classification map based on Sentinel 2 imagery. Various classification models exist, but it was opted to use the random forest algorithmn. This algorithmn was chosen since it has high dimensionality so can process the high amount of bands and reduce noise bias. It can also capture complex boundaries better which is case for the forests here on Samos.")
st.write("Most of the steps on this page is done with use of the Google Earth Engine API in Python.")

st.space(size="small")

st.subheader("Gathering imagery")
st.write("""
         For the gathering of the imagery the Sentinel 2 Surface Reflectance(SR) was used. The following parameters were used:
         - Start date: 01-03-2024
         - End date: 28-02-2025
         - Maximum cloud coverage: 40%
         """)

st.space(size="small")

st.subheader("Apply cloud mask")
st.write("It is almost impossible to get cloud free imagery and to still get a good temporal extent it's important to apply a cloud mask. For that the following tutorial was used: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless. This uses the S2 cloud probability dataset (s2cloudless) together with various thresholds to be able to mask out clouds and their shadows. These thresholds can be changed but for this project the default thresholds are used.")

st.space(size="small")

st.subheader("Compositing imagery")
st.write("""
         With the gathered and masked images composites can be made for each season. This is done to keep the processing time lower and there is no need for so many images for classification. A lot of differences can be found in the seasons so the images are composited based on the seasons. Following a median composite to remove any outliers found in the images.
         The seasons range from:
         - Spring: March till May 
         - Summer: June till August
         - Autumn: September till November
         - Winter: December till February
         """)

st.space(size="small")

st.subheader("Indices & Tassled Cap Transformation")
st.write("To be able to have a more accurate classification various indices and the tassled cap were calculated on the seasonal composites to be able to come to conclusions. Before being able to calculate these they the SWIR bands needed to be resampled to 10 meters.These are the following:")
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
        - Tassled Cap Transformation
            - This is a spectral analysis technique which converts spectral data into three components which singnify different properties.
            - Every sattelite has different coefficients which need to be used to apply the transformation correctly. For this project the following research was used to get these coefficients for Sentinel 2:
                - https://ieeexplore.ieee.org/document/8836649
            - The three components are as following:
                - To be able to calculate the components firstly the values need to be converted to true reflectance values. This is done by multiplying it by 0.0001
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
st.write("Since the classification model is a supervised model, features are needed to be able to distinguish between classes. In this way the model has something to base it's prediction on. Based on imagery the following features were chosen:")
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
st.write("Based on these features they could be split into a random 70:30 training split. Where 70% of the data will be used for the training of the random forest model and 30% will be used for the testing of the model.")

st.space(size="small")

st.subheader("Vegetation mask")
st.write("To only classify the forests a mask needs to be applied so only the forest will be classified. For this various summer indices with thresholds were used to be able to have the forests as the remaining pixels. They are as following:")
st.write("""
         Everything that fall under these thresholds will be masked out
         - NDVI: <0.70
         - NDBI: >0
         - NDWI: >-0.3
         - Greenness: <0.10
         - Brightness: <0.20 or >0.50
        """)

st.space(size="small")

st.subheader("Accuracy score")
st.write("After running the classification on the training samples different accuracy tests can be used based on the training and test samples.")
st.write("The proportion of classifications (total accuracy) that were correct is 0.9927 so 99.27%")
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
st.write("The random forest classification leads to this map as a final result:")
classification = leafmap.Map(
    zoom_control=True, 
    attribution_control=False,   
    draw_control=False,          
    measure_control=False,       
    locate_control=False,        
    scale_control=False  
)
classification.add_basemap("SATELLITE")
clas_file = "Data/Forest_classification.tif"
classification.add_raster(
            clas_file,
            layer_name="Forest classification",
            palette=["#00000000", "#006400", "#90ee90"]
)
legend_dict = {
    "Coniferous forest": "#006400",
    "Broadleaf forest": "#90ee90"
}
classification.add_legend(
    title="Forest types",
    legend_dict=legend_dict
)

classification.to_streamlit()

st.space(size="small")

st.subheader("Click below to view the final calculations to see the prediction of the total amount of carbon stored")
st.page_link(
    "pages/5_Total_carbon_stored.py",
    label="-> Carbon prediction"
)















