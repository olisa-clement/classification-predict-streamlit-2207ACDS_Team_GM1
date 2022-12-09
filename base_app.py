"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/countvect.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    # st.title("Tweet Classifer")
    # st.subheader("Climate change tweet classification")
    exc = False
    # Creating a side bar with company options
    nav = st.sidebar.radio("Navigation", ["Home", "About Us", "Contac Us"])
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    #options = ["Prediction", "Information"]
    #selection = st.sidebar.selectbox("Choose Option", options)

    # Create an empty list to populate predictions

    if nav == "Home":
        # Segment the page for various info
        first, second = st.columns(2)
        first.header("Hi! Welcome to DAB Analytics")
        first.text("We are excited to have you onboard.")
        second.image("archive.jpg", width=400)
        x, y = st.columns(2)
        y.text(
            "Want to share your thoughts on the climate change?\nCheck the Prediction button to get started")
        a, check = st.columns([2, 1])
        

        if check.checkbox("Prediction"):

            # Building out the predication page
            country, state = st.columns(2)
            country = country.text_input(
                "Country", placeholder="Enter your Country of Residence")
            state = state.text_input(
                "State", placeholder="Enter your Stete of Residence")
            empty, submit = st.columns([2, 1])

            if country != "" and state != "" and country.isdigit() == False and state.isdigit() == False:
                # Validating the data entered
                models = ['Logistic Regression', 'SVM', 'KNN',
                          'Naive Bayes', 'Random Forest', 'XGBoost']
                info, model_option = st.columns(2)
                info.info("Prediction with ML Models")
                ml_algo = model_option.selectbox(
                    "Please select a model", models)
                # Creating a text box for user input
                tweet_text = st.text_area(
                    "Enter Text", placeholder="Type Here")

                if ml_algo == "Logistic Regression":

                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/logistic_reg.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))


                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)




                if ml_algo == "SVM":

                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/SVM.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))

                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)




                if ml_algo == "KNN":
    
                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/KNN.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))

                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)



                
                if ml_algo == "Naive Bayes":
    
                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/NB.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))

                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)



                
                if ml_algo == "Random Forest":
    
                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/RF.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))
                        
                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)





                if ml_algo == "XGBoost":
    
                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/XGBoost.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        st.success(
                            "Text Categorized as: {}".format(prediction))

                        prediction_dict = {"Country": [country], "State": [state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv("resources/predictions.csv",mode='a',header = False,index= False)

                
                



        # Building out the "Information" page
        #if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        #st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

        # Building out the predication page
        # if selection == "Prediction":

    if nav == "About Us":

        logo, title = st.columns(2)

        logo.image("DAB.png", width=400)

        title.header("Who is DAB Analytics?")
        
        title.markdown("""<p style="text-align: left;">DAB Analytics is a team of passionate Data Scientists committed to delivering the highest quality Data Science products. We have worked in various industries solving seemingly impossible problems for clients all over the globe</p>""", unsafe_allow_html=True)

        st.header("Your Succes Team")
        st.markdown("Our staffs are world-class Data Scientist with significant years of experience and outstanding achievements in the world of data.")
        # Build the About Us structure
        olisa, joel, eze = st.columns(3)
        malik, ann, nnanna = st.columns(3)
        a, karabo, b = st.columns(3)

        # Populate the About Us page with the Company's staff info
        # Add staff images
        olisa.image("olisa.jpg", width=150, caption='Olisa, a team lead at DAB Analytics, comes from a technical background of Computer Engineering with 4+ years of experience in Data in the Financial Industry.')

        malik.image("malik.jpg", width=150, caption='Malik, a Machine Learning Engineer at DAB Analytics----- ')

        eze.image("Eze.jpg", width=150, caption='Eze, a Data Scientist at DAB Analytics with a tschniacal background in------')

        nnanna.image("Nnanna.jpg", width=150, caption='Nnanna, a Business Analyst at DAB Analytics-----')

        ann.image("Chiamaka.jpeg", width=150, caption='Ann, Strategy Manager at DAB Analytics------')

        joel.image("phot-joel.jpeg", width=150, caption='Joel hails from a rich academic background and holds PhD degree in Physics, and also a Machine Learning Specialist in DAB Analytics.')

        


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
