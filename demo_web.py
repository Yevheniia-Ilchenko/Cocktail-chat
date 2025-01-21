import streamlit as st
import requests

API_URL = "http://localhost:8000"

def get_response_from_api(query):
    try:
        response =requests.post(f"{API_URL}/chat", json={"query": query})
        if response.status_code == 200:
            return response.json().get("response", "No response from server.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"
    
def get_favorite_ingredients():
    try:
        response = requests.get(f"{API_URL}/favourite-ingredients")
        if response.status_code == 200:
            return response.json().get("favourite_ingredients", [])
        else:
            return []
    except Exception as e:
        return []
    
    
st.title("🍸🍹 Cocktail Chat 🍹🍸")

user_input = st.text_input("Enter your query:", 
                           placeholder="e.g., What are the 5 cocktails containing lemon?")
if st.button("Send"):
    if user_input.strip():
        with st.spinner("Getting response..."):
            response = get_response_from_api(user_input)
        st.markdown("Response:")
        st.write(response)
    else:
        st.warning("Please enter a query before submitting.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This is a simple chat application that integrates with a FastAPI backend for cocktail recommendations. 
    Use it to find cocktails, get recommendations based on your favorite ingredients, and more!
    """
)

st.sidebar.title("My favorites ingredients:")
favorites = get_favorite_ingredients()
if favorites:
    st.sidebar.write(",".join(favorites))
else:
    st.sidebar.write("No favorite ingredients yet.")
