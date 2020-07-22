# Description: This program detects if someone has diabetes using machine learning and python

# Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Open and display an image
image = Image.open('diabetes.png')
st.image(image, use_column_width = True)

# Create a title and a subtitle
st.write("""# Diabetes Detection""")

st.write("""## Detect if someone has diabetes using machine learning and python 👨‍⚕️""")

# Create a title and a subtitle at side bar
st.sidebar.title("Diabetes Detection")
st.sidebar.markdown("Detect if someone has diabetes using machine learning and python 👨‍⚕️")

# Author Acknowledgement
st.markdown("##### Created By Joseff Tan - [GitHub](https://github.com/ijoseff/Diabetes-Detection)")

# Get the data
df = pd.read_csv('diabetes.csv')

# Set a subheader
st.subheader('Data Information:')

# Show data source
st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/diabetes) is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.")

# Show the data as a table
st.dataframe(df)

# Set a subheader
st.subheader('Data Statistics:')

# Show statistics on the data
st.write(df.describe())

# Set a subheader
st.subheader('Data Visualization')

# Show the data as a chart
chart = st.bar_chart(df)


# Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split the data set into 80% training and 20% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Get the feature input from the user
def get_user_input():
	Pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 3)
	Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
	Blood_Pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
	Skin_Thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
	Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
	BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
	DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
	Age = st.sidebar.slider('Age', 21, 81, 29)

	# Store dictionary into variable
	user_data = {'Pregnancies': Pregnancies,
				 'Glucose': Glucose,
				 'Blood Pressure': Blood_Pressure,
				 'Skin Thickness': Skin_Thickness,
				 'Insulin': Insulin,
				 'BMI': BMI,
				 'Diabetes Pedigree Function': DiabetesPedigreeFunction,
				 'Age': Age}

	# Transform the data into a data frame
	features = pd.DataFrame(user_data, index = [0])
	return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)


# Create and train the model
RandomForestClassifier = RandomForestClassifier(n_estimators= 100, max_depth= 200, criterion = 'entropy', random_state = 0)
RandomForestClassifier.fit(X_train, Y_train)

# Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test) * 100)) + '%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)
st.warning('Caution: The model needs more improvment!')

# Set a subheader and display the classification
st.subheader('Classification:')
st.markdown('0 = FALSE | 1 = TRUE')
st.write(prediction)
