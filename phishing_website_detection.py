import streamlit as st
import machine_learning as ml
import feature_extraction as fe 
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

# col1, col2 = st.columns([1, 3])
st.title("Final Year Project 2019-2023 Batch,Madanapalle Institute of Technology and Science")
st.title('By using Deep Learning algorithms to detect whether the website is real or fake')
#st.title('Phishing Website Detection using Deep Learning')
#st.write('This ML-based app is developed for educational purposes. Objective of the app is detecting phishing websites only using content data. Not URL!'
#         ' You can see the details of approach, data set, and feature set if you click on _"See The Details"_. ')


with st.expander("PROJECT DETAILS"):
    st.subheader('Methedology')
    st.write('Deep learning techniques are very efficient for natural language and image classification.Here,the convolutional neural network (CNN) and the long short-term memory (LSTM) algorithms were used to build the intelligent phishing detection system (IPDS).')
    st.write('We created our own data set and defined features, some from the literature and some based on manual analysis. '
             'We used requests library to collect data, BeautifulSoup module to parse and extract features. ')
    st.write('The data sets and source code are available in my Github link:')
    st.write('https://github.com/MADANI99/Detecting-the-Phishing-Website')

    st.subheader('Data set')
    st.write('We used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 25235 websites  --> Among those **_18674_ are legitimate** websites and **_6561_ phishing** are websites')
    st.write('We recently updated the data set in the month of November 2022.')

    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 150 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #

    st.write('Dataframe --> Features + URL + Label')
    st.markdown('label=1 for phishing')
    st.markdown('label=0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))


    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader('Features')
    st.write('Here firstly we used only content-based features. We didn\'t use url-based faetures like length of url etc.'
             'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

    st.subheader('Results')
    st.write('We used 7 different deep learning classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    st.table(ml.df_results)
    st.write('NB --> Gaussian Naive Bayes')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')

with st.expander('PHISHING URLs EXAMPLE:'):
    st.write('_https://tqybxmkeie.duckdns.org/_')
    st.write('_http://www.vsvesvccaaveceia.ceooaesc.nhusiwa.icu/k7OIMyJhEU/page1.php_')
    st.write('_https://ncarkwugwp.duckdns.org/_')
    st.caption('One more thing, This Phishing web pages have short life cycle!. So the above examples are mentioned for demo purpose. If these are not working properly we need to update the examples')

choice = st.selectbox("Please select your deep learning model",
                 [
                     'Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                     'AdaBoost', 'Neural Network', 'K-Neighbours'
                 ]
                )

model = ml.nb_model

if choice == 'Gaussian Naive Bayes':
    model = ml.nb_model
    st.write('GNB model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model is selected!')
elif choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')
elif choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AB model is selected!')
elif choice == 'Neural Network':
    model = ml.nn_model
    st.write('NN model is selected!')
else:
    model = ml.kn_model
    st.write('KN model is selected!')


url = st.text_input('Enter the URL')
# check the url is valid or not
if st.button('Check!'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            print("HTTP connection was not successful for the URL: ", url)
            st.warning("Warning! This web page looks like PHISHING!")
            st.snow()
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page is legitimate!")
                st.balloons()
            else:
                st.warning("Warning! This web page looks like PHISHING!")
                st.snow()

    except re.exceptions.RequestException as e:
        print("--> ", e)





