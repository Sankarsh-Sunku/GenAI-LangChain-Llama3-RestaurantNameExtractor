import streamlit as st
import langchain_helper

st.title("Restaurant Name and Menu Generator")

cuisine =  st.sidebar.selectbox("Pick a Cuisine", ('Indian', 'Mexican', 'Italian', 'French'))


if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurantName'])
    menu_items = response['menuItems']
    st.header("**Menu Items**")
    st.write(menu_items)

