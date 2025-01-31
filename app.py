import streamlit as st
from agent import initiate_chat
import asyncio


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Software Architecture Agent", initial_sidebar_state="expanded")
    st.title("Software Architecture AI Bot")
    user_question = st.text_input("Please specify you application requirements")

    analyze_button = st.button("Ask", help="Click to start the software design")
    if analyze_button:
        st.write(initiate_chat(user_question))




