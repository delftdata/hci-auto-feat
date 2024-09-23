from streamlit_pandas_profiling import st_profile_report
from datetime import datetime, time
import numpy as np
import altair as alt
import pandas as pd
import ydata_profiling
import streamlit as st 


### Day 2 

st.header('st.button')

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye!')

### Day 6 

st.header('st.write')

st.write('Hello, *World!* :sunglasses:')

st.write(1234)

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
st.write(df)

st.write('Below is a dataframe:', df, "Above is a dataframe")

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)

### Day 8

st.header('st.slider')

st.subheader('Slider')

age = st.slider('How old are you?', 0, 130, 25)
st.write('I am', age, 'year old')

st.subheader('Range Slider')

values = st.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
    )
st.write('Values:', values)

st.subheader('Range time slider')
appointment = st.slider(
    'Schedule your appointment:',
    value=(time(11, 30), time(12, 45))
)
st.write("You're scheduled for:", appointment)

st.subheader('Datetime slider')

start_time = st.slider(
    "When do you start?",
    value=datetime(2020, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm"
)
st.write("Start time:", start_time)

### Day 9 

st.header('st.line_chart')

df = pd.DataFrame(
    np.random.randn(100, 2),
    columns=["a", "b"]
)
st.line_chart(df)

### Day 10 

st.header("st.selectbox")

options = ('Red', "Blue", "Yellow", "Green", "Orange", "Pink")
color = st.selectbox("What is your favourite colour?", options=options)
st.write("My favourite colour is:", color)

### Day 11 

st.header("st.multiselect")

options = st.multiselect(
    "What are your favourite colours?",
    options=["Green", "Blue", "Yellow", "Red"],
    default=["Yellow", "Red"]
)
st.write("You selected:", options)

### Day 12

st.header("st.checkbox")

st.write("What would you like to order?")

icecream = st.checkbox('Ice cream')
coffee = st.checkbox('Coffee')
cola = st.checkbox('Coca Cola')

if icecream:
    st.write("Great! Here is your :icecream:")

if coffee:
    st.write("Ok, there you go :coffee:")

if cola:
    st.write("Enjoy your cola!")

### Day 14

st.header("Components")

# df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
# pr = df.profile_report()
# st_profile_report(pr)

### Day 15 

st.header("st.latex")

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

### Day 16 

st.title('Customizing the theme of Streamlit apps')

st.write('Contents of the `.streamlit/config.toml` file of this app')

st.code("""
[theme]
primaryColor="#F39C12"
backgroundColor="#2E86C1"
secondaryBackgroundColor="#AED6F1"
textColor="#FFFFFF"
font="monospace"
""")

number = st.sidebar.slider('Select a number:', 0, 10, 5)
st.write('Selected number from slider widget is:', number)

### Day 17 

st.title("st.secrets")
# st.write(st.secrets['message'])


