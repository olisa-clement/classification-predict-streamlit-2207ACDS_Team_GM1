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
news_vectorizer = open("resources/tfidfvect_team1.pkl", "rb")
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
    nav = st.sidebar.radio("Navigation", ["Home", "About Us"])
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    # options = ["Prediction", "Information"]
    # selection = st.sidebar.selectbox("Choose Option", options)

    # Create an empty list to populate predictions

    if nav == "Home":
        # Segment the page for various info
        first, second = st.columns(2)
        first.header("Hi! Welcome to DAB Analytics")
        first.markdown("""<p style="text-align: justify; font-size:12px">We are excited to have you onboard.\nIn DAB Analytics, we provide solutions to world problems using data. Climate change has been a serious concern to the world at large and we want to help solve one of the biggest challenge of Man kind through the products we sell to our valued customers. Would you want to join us and save the planet?\n Awesome! Check the Predicton button to get started. We want to know what your thoughts are on man-made climate change.</p>""", unsafe_allow_html=True)
        second.image("archive.jpg", width=400)
        x, y = st.columns(2)
        y.text(
            "Want to share your thoughts on the climate change?\nCheck the Prediction button to get started")
        a, check = st.columns([2, 1])

        if check.checkbox("Prediction"):

            st.markdown("Please enter your country and state, then press Enter")

            # Building out the predication page
            country, state = st.columns(2)
            country = country.text_input(
                "Country", placeholder="Enter your Country of Residence (Only text)")
            state = state.text_input(
                "State", placeholder="Enter your Stete of Residence (Only text)")
            empty, submit = st.columns([2, 1])

            if country != "" and state != "" and country.isdigit() == False and state.isdigit() == False:
                # Validating the data entered
                models = ['Logistic Regression',
                          'Naive Bayes', 'Random Forest']
                info, model_option = st.columns(2)
                info.info("Information About The Selected ML Model")
                
                ml_algo = model_option.selectbox(
                    "Please select a model", models)
                # Creating a text box for user input
                tweet_text = st.text_area(
                    "Tell us your thoughts on Climate Change", placeholder="Type Here")

                if ml_algo == "Logistic Regression":

                    info.text("Logistic Regression has an Accuracy of 72%")

                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/logistic_reg_team1.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        if prediction == 1:
                            st.success(
                                "Pro: You believe in man-made climate change")

                        if prediction == 2:
                            st.success(
                                "News: This is a factual news about climate change")

                        if prediction == 0:
                            st.success(
                                "Neutral: You have a neutral opinion with regards to man-made climate change")

                        if prediction == -1:
                            st.success(
                                "Anti: You do not believe in man-made climate change")

                        prediction_dict = {"Country": [country], "State": [
                            state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv(
                            "resources/predictions.csv", mode='a', header=False, index=False)

                if ml_algo == "Naive Bayes":

                    info.text("Naive Bayes has an Accuracy of 62%")

                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/NB_team1.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        if prediction == 1:
                            st.success(
                                "Pro: You believe in man-made climate change")

                        if prediction == 2:
                            st.success(
                                "News: This is a factual news about climate change")

                        if prediction == 0:
                            st.success(
                                "Neutral: You have a neutral opinion with regards to man-made climate change")

                        if prediction == -1:
                            st.success(
                                "Anti: You do not believe in man-made climate change")
                        prediction_dict = {"Country": [country], "State": [
                            state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv(
                            "resources/predictions.csv", mode='a', header=False, index=False)

                if ml_algo == "Random Forest":

                    info.text("Random Forest has an Accuracy of 81%")

                    if st.button("Classify"):
                        # Transforming user input with vectorizer
                        vect_text = tweet_cv.transform([tweet_text]).toarray()
                        # Load your .pkl file with the model of your choice + make predictions
                        # Try loading in multiple models to give the user a choice
                        predictor = joblib.load(
                            open(os.path.join("resources/RF_team1.pkl"), "rb"))
                        prediction = predictor.predict(vect_text)

                        # When model has successfully run, will print prediction
                        # You can use a dictionary or similar structure to make this output
                        # more human interpretable.
                        if prediction == 1:
                            st.success(
                                "Pro: You believe in man-made climate change")

                        if prediction == 2:
                            st.success(
                                "News: This is a factual news about climate change")

                        if prediction == 0:
                            st.success(
                                "Neutral: You have a neutral opinion with regards to man-made climate change")

                        if prediction == -1:
                            st.success(
                                "Anti: You do not believe in man-made climate change")
                        prediction_dict = {"Country": [country], "State": [
                            state], "Predictions": [prediction]}
                        prediction_df = pd.DataFrame(prediction_dict)
                        prediction_df.to_csv(
                            "resources/predictions.csv", mode='a', header=False, index=False)

        # Building out the "Information" page
        # if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        # st.markdown("Some information here")

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
        st.markdown(
            "Our staffs are world-class Data Scientist with significant years of experience and outstanding achievements in the world of data.")
        # Build the About Us structure
        olisa, joel, eze = st.columns(3)
        malik, ann, nnanna = st.columns(3)
        a, karabo, b = st.columns(3)

        # Populate the About Us page with the Company's staff info
        # Add staff images
        olisa.image("olisa.jpg", width=150, caption='Olisa, a team lead at DAB Analytics, comes from a technical background of Computer Engineering with 4+ years of experience in Data in the Financial Industry.')

        malik.image("malik.jpg", width=150,
                    caption='Malik, a Machine Learning Engineer at DAB Analytics----- ')

        eze.image("Eze.jpg", width=150,
                  caption='Eze, a Data Scientist at DAB Analytics with a tschniacal background in------')

        nnanna.image("Nnanna.jpg", width=150,
                     caption='Nnanna, a Business Analyst at DAB Analytics-----')

        ann.image("Chiamaka.jpeg", width=150,
                  caption='Ann, Strategy Manager at DAB Analytics------')

        joel.image("phot-joel.jpeg", width=150,
                   caption='Joel hails from a rich academic background and holds PhD degree in Physics, and also a Machine Learning Specialist in DAB Analytics.')

        karabo.image("karabo.jpg", width=150,
                     caption='Karabo, a Machine Learning Engineer at DAB Analytics------')


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
