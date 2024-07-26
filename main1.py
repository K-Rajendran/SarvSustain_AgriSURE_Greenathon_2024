import os
import json
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

#Page configuration
st.set_page_config(
    page_title="SurvSustain",layout="wide", page_icon="", initial_sidebar_state="expanded"
)

#--------------------Crop Recommendation--------------------------
RF_Model_pkl=pickle.load(open('F:\\Projects\\Plant Disease Detection\\CropRecommendation\\RF.pkl','rb'))

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction1 = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction1

#--------------------Crop Yield Prediction--------------------------
model=joblib.load('F:\\Projects\\Plant Disease Detection\\CropYield\\model.pkl')
sc=joblib.load('F:\\Projects\\Plant Disease Detection\\CropYield\\sc.pkl')
pf=joblib.load('F:\\Projects\\Plant Disease Detection\\CropYield\\pf.pkl')

df_final=pd.read_csv('F:\\Projects\\Plant Disease Detection\\CropYield\\test.csv')
df_main=pd.read_csv('F:\\Projects\\Plant Disease Detection\\CropYield\main.csv')

def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df
def prediction(input):
    categorical_col=input[:2]
    input_df=pd.DataFrame({'average_rainfall':input[2],'presticides_tonnes':input[3],'avg_temp':input[4]},index=[0])
    input_df1=df_final.head(1)
    input_df1=input_df1.iloc[:,3:]
    true_columns = [f'Country_{categorical_col[0]}',f'Item_{categorical_col[1]}']
    input_df2= update_columns(input_df1, true_columns)
    final_df=pd.concat([input_df,input_df2],axis=1)
    final_df=final_df.values
    test_input=sc.transform(final_df)
    test_input1=pf.transform(test_input)
    predict=model.predict(test_input1)
    result=(int(((predict[0]/100)*2.47105) * 100) / 100)
    return (f"The Production of Crop Yields:- {result} quintel/acers yield Production. "
            f"That means 1 acers of land produce {result} quintel of yield crop. It's all depend on different Parameter like average rainfall, average temperature, soil and many more.")

#--------------------Plant Disease Detection--------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path1 = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model1 = tf.keras.models.load_model(model_path1)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

#-----------------------------Sidebar--------------------------
with st.sidebar:

    st.markdown("<h1 style='text-align: center;'>🌿SARVSUSTAIN🍁</h1>", unsafe_allow_html=True)
    selected = option_menu('NAVIGATION', [
        'Home',
        'Idea Brief',
        'Tech Stack',
        'Crop Recommendation',
        'Crop Yield Prediction',
        'Plant Disease Detection'
    ],
        icons=['bi bi-house-fill','bi bi-info-square','bi bi-stack','bi bi-cloud-sleet-fill','bi bi-brightness-high-fill','bi bi-flower1'],
        menu_icon="house-heart-fill", default_index=0,
        styles={
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"}
        }
    )
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    #st.sidebar.markdown("<h5 style='text-align: center;'>PLANT_DISEASE_DETECTION_WEBPAGE | MAY 2024 | JAI GANESH | HARIHARAN | RAJENDRAN</h3>", unsafe_allow_html=True)

#--------------------------------Home page-----------------------------------
if selected == 'Home':
    image_path = "F:\Projects\Plant Disease Detection\sarvsustain-high-resolution-logo.png"
    st.image(image_path,use_column_width=True)
    
#-------------------------------Idea Section---------------------------------------
if selected == 'Idea Brief':
    st.markdown("<h1 style='text-align: center; color: #000080;'>IDEA BRIEF</h1>", unsafe_allow_html=True)
    st.markdown("_________")
    st.write(
        "1. Combines sensor-based data collection, IoT, machine learning, and deep learning for hyper-local decision making. "
    )
    st.write(
        "2. Provides real-time insights on soil moisture, temperature, and plant health through affordable sensors and secure IoT networks. "
    )
    st.write(
        "3. Optimizes resource allocation, predicts crop yields, and reduces pesticide dependence with AI-powered analytics. "
    )
    st.write(
        "4. Predicts weather patterns, disease outbreaks, and pest infestations using machine learning algorithms for proactive crop protection. "
    )
    st.write(
        "5. Offers prescriptive analysis and suggestions on water usage, fertilizer usage, and growth prediction with Large Language Models (LLMs) and Retrieval-Augmented Generative (RAG) models. "
    )   
    st.write(
        "6. Incorporates demand-supply forecasting to enable informed decisions on crop selection and pricing based on historical sales and price data. "
    )
    st.write(
        "7. Fosters collaboration, knowledge sharing, and sustainable agricultural practices through personalized recommendations, benchmarking, and community insights. "
    )
    st.write(
        "8. Leads to increased yields, improved resource management, and a more sustainable agricultural system. "
    )
    st.write(
        "9. Data-driven insights assist policymakers in making informed decisions on agricultural policies, subsidies, and regulations for a supportive ecosystem and positive impact. "
    )

#------------------------------------Tech Stack--------------------------------
if selected == 'Tech Stack':
    st.markdown("<h1 style='text-align: center; color: #000080;'>TECH STACK</h1>", unsafe_allow_html=True)
    st.markdown("_________")
    st.subheader("1. CENTRAL:")
    st.markdown("- Cloud Infrastructure (Leading Providers) with Containerization & Serverless Computing")
    st.subheader("2. DATA ACQUISTION:")
    st.markdown("- Affordable IoT Sensors (Soil Moisture, Temperature, Humidity, etc.)")
    st.markdown("- Image Acquisition (Cameras)")
    st.markdown("- Communication: Wi-Fi, Bluetooth, LoRaWAN, Cellular Networks")
    st.subheader("3. DATA PROCESSING AND ANALYTICS:")
    st.markdown("- Cloud-based Databases (Storage)")
    st.markdown("- Machine Learning & Deep Learning Frameworks (TensorFlow, PyTorch, Keras)")
    st.markdown("- Libraries (scikit-learn, OpenCV, Pillow)")
    st.write(
        "a. Image Classification"
    )
    st.write(
        "b. Object Detection"
    )
    st.write(
        "c. Time Series Forecasting (Weather, Disease, Pests)"
    )
    st.subheader("4. INSIGHTS AND RECOMMENDATIONS:")
    st.markdown("- Large Language Models (LLMs)")
    st.markdown("- Retrieval-Augmented Generative (RAG) models")
    st.markdown("- Real-time Insights (Soil, Temperature, Plant Health)")
    st.markdown("- Resource Allocation Optimization")
    st.markdown("- Crop Yield Prediction")
    st.markdown("- Reduced Pesticide Dependence")
    st.markdown("- Personalized Recommendations")
    st.markdown("- Benchmarking")
    st.markdown("- Community Insights")
    st.subheader("5. VISUALIZATION AND USER INTERFACE:")
    st.markdown("- Data Visualization Tools (Dashboards, BI)")
    st.markdown("- Front-end & UI (React, Material Design, Bootstrap, Tailwind CSS)")
    st.markdown("- Mobile App Development (iOS, Android, React Native)")
    st.subheader("6. SECURITY:")
    st.markdown("- SSL/TLS Encryption")
    st.markdown("- Firewalls")
    st.markdown("- Intrusion Detection Systems")
    st.markdown("- Data Protection & Regulatory Compliance")

#----------------------------------Crop recommendation---------------------------------------
if selected == 'Crop Recommendation':
    st.markdown("<h1 style='text-align: center; color: #000080;'>CROP RECOMMENDATION</h1>", unsafe_allow_html=True)
    st.markdown("_________")
    nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs=[[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]                                               
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.button("Predict"):
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction1 = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction1[0]}")

#-------------------------------Crop Yield Prediction---------------------------
if selected == 'Crop Yield Prediction':
    st.markdown("<h1 style='text-align: center; color: #000080;'>CROP YIELD PREDICTION</h1>", unsafe_allow_html=True)
    st.markdown("_________")
    country= st.selectbox("Type or Select a Country from the Dropdown.",df_main['area'].unique()) 
    crop= st.selectbox("Type or Select a Crop from the Dropdown.",df_main['item'].unique()) 
    average_rainfall=st.number_input('Enter Average Rainfall (mm-per-year).',value=None)
    presticides=st.number_input('Enter Pesticides per Tonnes Use (tonnes of active ingredients).',value=None)
    avg_temp=st.number_input('Enter Average Temperature (degree celcius).',value=None)
    input=[country,crop,average_rainfall,presticides,avg_temp]
    result=''
    if st.button('Predict',''):
        result=prediction(input)
    temp='''
     <div style='background-color:navy; padding:8px'>
     <h1 style='color: gold  ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp,unsafe_allow_html=True)
    
#-------------------------------Plant Disease Prediction----------------------------
if selected == 'Plant Disease Detection':
    st.markdown("<h1 style='text-align: center; color: #000080;'>PLANT DISEASE RECOGNITION</h1>", unsafe_allow_html=True)
    st.markdown("_________")
    st.header("Disease Recognition")
    uploaded_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
        
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        
        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            #Predict button
            if(st.button("Predict")):
                st.snow()
                st.write("Our Prediction")
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model1, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')