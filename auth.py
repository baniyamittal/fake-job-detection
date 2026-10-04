import streamlit as st
from database import register_user, login_user

def register_page():
    st.title("Create Account")
    st.write("Register to access the Fake Job Detection System")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register", use_container_width=True):
        if not name or not email or not password or not confirm_password:
            st.warning("Please fill in all fields.")

        elif password != confirm_password:
            st.error("Passwords do not match.")

        elif len(password) < 6:
            st.warning("Password must contain at least 6 characters.")

        else:
            success, message = register_user(
                name,
                email,
                password
            )

            if success:
                st.success(message)
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(message)

def login_page():
    st.title("Welcome Back")
    st.write("Login to your Fake Job Detection account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login", use_container_width=True):
        if not email or not password:
            st.warning("Please enter email and password.")

        else:
            success, user = login_user(
                email,
                password
            )

            if success:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.user_name = user[1]
                st.session_state.user_email = user[2]
                st.session_state.page = "dashboard"
                st.rerun()

            else:
                st.error("Invalid email or password.")

    st.write("")

    if st.button("Create New Account"):
        st.session_state.page = "register"
        st.rerun()