# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 17:25:35 2022

@author: Calvin
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.sidebar.header("Input your values:")
def InputData():
    DailyTime = st.sidebar.slider("Daily Time Spent on Site",min_value=32.6,max_value=91.43,value=32.6)
    Age = st.sidebar.slider("Age",min_value=19,max_value=61,value=19)
    AreaIncome = st.sidebar.slider("Area Income",min_value=13996.5,max_value=79484.8,value=14000.0)
    DailyInternet = st.sidebar.slider("Daily Internet Usage",min_value=104.78,max_value=269.96,value=105.0  )
    Gender = st.sidebar.selectbox("Gender", ("Male","Female"))
    
    def GenderChange(x):
        if x == "Male":
            y = 1
        else:
            y=0
        return y
    
    GenderCode = GenderChange(Gender)
    data = {
            "Daily Time Spent on Site":[DailyTime],
            'Age':[Age],
            'Area Income':[AreaIncome],
            'Daily Internet Usage':[DailyInternet],
            'Male':[GenderCode]
            }
    
    df = pd.DataFrame(data)
    return df

df1 = InputData()

ad_data = pd.read_csv('advertising.csv')

st.markdown("<h5 style='text-align: left; '>By: Calvin Hartono</h5>", unsafe_allow_html=True)

st.write("The dataset used in this portfolio could be accessed through [here](https://drive.google.com/file/d/1pCnCb5n5_2h66O9e4Tf0crbTNFnsIpHI/view?usp=sharing). This project was inspired from Jose Portilla's course at Udemy - [Data Science and Machine Learning](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/). This project was created with Python and the libraries used are: streamlit, pandas, plotly, seaborn, scikit-learn. The content of the dataset could be seen below:")
st.write(ad_data)
st.write("The dimension of the data: " + str(len(ad_data)) + ' rows and ' + str(len(ad_data.columns)) + ' columns')

st.markdown("""---""")

st.write("")
st.markdown("<h3 style='text-align: left; '>Exploratory Data Analysis (EDA)</h3>", unsafe_allow_html=True)
st.write("To decide the X variables that could be used to predict the number of click on ad, i did some Exploratory Data Analysis (EDA) using Plotly and Seabon Libraries with different kinds of visualizations. They are histogram, scatterplot, jointplot, dan countour diagram. From the EDA, i could find the relation and correlation between some of these variables.")

histo = px.histogram(data_frame=ad_data['Age'],nbins=30,title="Age Distribution",labels={'value':'Age'},color_discrete_sequence=['royalblue'], height=500, width=750)
histo.update_layout(title_x=0.5, bargap=0.1)
st.plotly_chart(histo)
st.write('From the histogram of age distribution, we could see that the distribution of age in the dataset is quite normally distributed around 30-35 years old')

st.markdown("""---""")

fig = px.scatter(ad_data, x='Age', y='Area Income', marginal_y="histogram", marginal_x="histogram", title='Distribution of Age and Area Income',width=750, height=750, color_discrete_sequence=['darkblue'])
fig.update_layout(title_x=0.5, bargap=0.1)
st.plotly_chart(fig)
st.write('From the jointplot, we could see the relationship between age and area income. We could see that the diagram is quite scattered but there is still a trend, in which the follow the increasing of age from 20 to 40 years old, the income are increasing and it starts to decrease after 40 years old')

st.markdown("""---""")

fig2 = px.density_contour(ad_data, x="Age", y="Daily Time Spent on Site", marginal_x="histogram", marginal_y="histogram",color_discrete_sequence=['darkred'],nbinsx=15, nbinsy=15, title='Distribution of Age and Daily Time Spent on Site', width=750, height=750)
fig2.update_layout(title_x=0.5, bargap=0.1)
st.plotly_chart(fig2)
st.write('From the contour diagram, we could see that majority that spends most time on site is people within the age of 25 years old until 35 years old. The amounts of time they spent are around 75-90 minutes per day ')

st.markdown("""---""")

fig3 = px.scatter(ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage', marginal_y="histogram", marginal_x="histogram",color_discrete_sequence=['green'],width=750, height=750, title='Distribution of Daily Internet Usage and Daily Time Spent on Site')
fig3.update_layout(title_x=0.5, bargap=0.1)
st.plotly_chart(fig3)
st.write("From the jointplot, we could see there are two clusters, the cluster of high daily time spent on site with high daily internet usage and cluster of low daily time spent of site with low daily internet usage. To explore it further, i did jointplot for every single numerics in the dataset with Seaborn which can be seen in the next plot.")

st.markdown("""---""")

st.pyplot(sns.pairplot(ad_data,hue='Clicked on Ad', height=2))
st.write('From the pairplot, we could see the correlation and relation between each of the numeric variables with the hue of clicked on ad. We could see there are two clusters in each of the plot, so it could be concluded that these variables could affect the numbers of click on ad')

st.markdown("""---""")

st.write("")
st.markdown("<h3 style='text-align: left; '>Logistic Regression</h3>", unsafe_allow_html=True)
st.write('For the next step, i created the prediction model for whether a person with certain conditions would click on ad or not. The dataset was divided into 2 kinds of variables. The X variables (independent) were daily time spent of site, age, area income, daily internet usage, and male. Meanwhile, the y variabel (dependent) was the clicked on ad which is the variable i would like to predict.')

st.write('The dataset is divided into training data and test data. The size of the training data i used is 67% and the size of the test data is 33%. The purpose of dividing the data into training data and test data is to avoid overfitting and underfitting model. The random state used is 1.')

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

st.write('The classification report for prediction of the test data could be seen below:')
st.text(classification_report(y_test,predictions))

st.write('')
st.write('')
st.write('The confusion metrix for prediction of the test data could be seen below:')
st.text(confusion_matrix(y_test,predictions))

st.markdown("""---""")

st.write('')
st.markdown("<h3 style='text-align: left; '>Prediction:</h3>", unsafe_allow_html=True)
if (logmodel.predict(df1)) == 1:
    st.markdown("<h5 style='text-align: left; color: red; '>You clicked on ad</h5>", unsafe_allow_html=True)
else:
    st.markdown("<h5 style='text-align: left; color: red; '>You didn't click on ad</h5>", unsafe_allow_html=True)

st.write('')
st.markdown("""---""")
