import streamlit as st

# Application title
st.title('My First Streamlit Application')

# Introductory text
st.write('Welcome to my interactive application.')

# Add a button
if st.button('Click here'):
    st.write('The button has been clicked.')

# Add a text-input field
user_input = st.text_input('Enter your name and press Enter:', '')
if user_input:
    st.write(f'Hello, {user_input}.')

# Add a section for a simple calculator
st.write('### Simple Calculator')

# Numeric inputs
num1 = st.number_input('Enter the first number', value=0)
num2 = st.number_input('Enter the second number', value=0)

# Button for adding the numbers
if st.button('Add'):
    result = num1 + num2
    st.write(f'The result of the addition is: {result}')

# Numeric inputs
num3 = st.number_input('Enter the first number', value=0, key='third')
num4 = st.number_input('Enter the second number', value=0, key='fourth')

# Button for multiplying the numbers
if st.button('Multiply'):
    result = num3 * num4
    st.write(f'The result of the multiplication is: {result}')

# Closing message
st.write('You may include any kind of content in this application, including calculations, visualizations, and user interaction.')
