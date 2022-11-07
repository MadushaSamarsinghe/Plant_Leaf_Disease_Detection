import numpy as np
import streamlit as st
import cv2 as cv
from keras.models import load_model

model = load_model('./dataset/disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

st.title("Plant Disease Detection")
st.text("Digital Image Processing - Final Project | SE 3071 | 2022 | MLB_IM_WD_06")
st.subheader("Introduction")
st.markdown(
    """
    We are surrounded by the Nature since birth to the death. 
    Plants are one of the most important living beings that are connected 
    with our day-to-day life. For respiration, Food supply, Water cycle, 
    the existence of the plants are really important which shows, 
    all living beings including humans depend on entirely on plants 
    for clean air and a livable climate as well as for food, medicines, 
    materials, and well-being. Throughout the years, people who belong to 
    different cultures around the world, learned to identify plants 
    especially by the structure, odor of the plant leaves. This accumulated 
    traditional knowledge of plant leaves has allowed the ancient people to 
    survive in diverse environments for thousands of years.
    """
)
st.markdown(
    """
    This traditional knowledge was passed from generation to generation 
    orally which influenced naming he plants and grouping the plants by 
    identifying the leaf. But with the development of modern world, the 
    space for the nature and the time to connect with the nature for the 
    younger generation are limited. Hence the identification of plant 
    leaves using the computer based technology being used in today world. 
    Not only plant leaves, but identification of diseases in plant leaves 
    is also important as all living organisms on Earth rely on the process 
    of photosynthesis for food energy and oxygen which is done by the 
    leaves of the plants.
    """
)
st.subheader("Quick Overview of Project")
st.markdown(
    """
    Identification of various kind of diseases in plant leaves is the main purpose of this
    project which is something really helpful when it comes to Agro world.
    This will greatly helpful for farmers and in other agricultural people to
    identify diseases of plants with less effort and often accurate information.
    Here is the our approach to that Plant Leaf disease detecting System. Currently,
    it's capable of identifying leaf diseases like Common Rust in Corn leaves, Early 
    Blight in Potato leaves and Bacterial Spot in Tomoato leaves. for that, we are using
    a Pre-trained Custom CNN Model using nearly 900 of Dataset. 
    """
)
st.subheader("Team Members")
st.markdown(
    """
        - D.K.S.L Kumara - IT20659080 (Leader)
        - S.K.M.S.Samarasinghe - IT20654108
        - RAJA .R.K.K - IT20667146
        - J. Sanjeevarajah - IT19095868
    """,
)
st.subheader("Tech Stack used in this Project")
st.text("OpenCV | Matplotlib | Streamlit | Tenserflow | Pandas | Numpy | Sklearn")
st.subheader("How it works ?")
st.markdown(
    """
    """
)
st.markdown(
    """
    *****
    """
)











st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
plant_image = st.file_uploader("Pick an Image with disease . . .", type="jpg")
submit = st.button('Predict the Disease')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        #Resizing the image
        opencv_image = cv.resize(opencv_image, (256,256))

        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
