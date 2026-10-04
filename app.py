import streamlit as st
import pandas as pd
from src.predict import predict_job
from database import (
    register_user,
    login_user,
    save_prediction,
    get_prediction_history,
    get_prediction_statistics
)

st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🛡️",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 18px;
    border-radius: 12px;
}

h1 {
    font-weight: 700;
}

h2 {
    font-weight: 650;
}

h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


def login_page():

    st.title("🛡️ Fake Job Detection System")
    st.subheader("Welcome Back")

    st.write(
        "Login to access the Fake Job Detection Dashboard."
    )

    email = st.text_input("Email")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        if not email or not password:
            st.warning(
                "Please enter email and password."
            )

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

                st.error(
                    "Invalid email or password."
                )

    st.divider()

    if st.button(
        "Create New Account",
        use_container_width=True
    ):

        st.session_state.page = "register"
        st.rerun()


def register_page():

    st.title("🛡️ Fake Job Detection System")
    st.subheader("Create Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password"
    )

    if st.button(
        "Register",
        use_container_width=True
    ):

        if not name or not email or not password or not confirm_password:

            st.warning(
                "Please fill in all fields."
            )

        elif password != confirm_password:

            st.error(
                "Passwords do not match."
            )

        elif len(password) < 6:

            st.warning(
                "Password must contain at least 6 characters."
            )

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

    st.divider()

    if st.button(
        "Back to Login",
        use_container_width=True
    ):

        st.session_state.page = "login"
        st.rerun()


def dashboard_page():

    st.title("🏠 Dashboard")

    total, genuine, fraudulent, fraud_rate = get_prediction_statistics(
        st.session_state.user_id
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jobs Analyzed",
        total
    )

    col2.metric(
        "Genuine Jobs",
        genuine
    )

    col3.metric(
        "Fraudulent Jobs",
        fraudulent
    )

    col4.metric(
        "Fraud Detection Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            ### 🔍 Fake Job Detection

            The system analyzes job postings using:

            • Job title  
            • Company profile  
            • Description  
            • Requirements  
            • Benefits  
            • Location  
            • Department  
            • Employment type  
            • Required experience  
            • Required education  
            • Industry  
            • Job function  
            • Telecommuting  
            • Company logo  
            • Screening questions
            """
        )

    with col2:

        st.success(
            """
            ### 🤖 Machine Learning

            The system combines:

            • TF-IDF text features  
            • Structured categorical features  
            • Binary features  
            • Logistic Regression  
            • Naive Bayes  
            • Random Forest  
            • Linear SVM  

            The selected model is a calibrated Linear SVM.
            """
        )

    st.divider()

    st.subheader("Recent Predictions")

    history = get_prediction_history(
        st.session_state.user_id
    )

    if history:

        recent = history[:5]

        df = pd.DataFrame(
            recent,
            columns=[
                "ID",
                "Job Title",
                "Prediction",
                "Risk Level",
                "Risk Score",
                "Date"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No predictions yet. Go to Detect Job to analyze your first posting."
        )


def detect_job_page():

    st.title("🔍 Fake Job Detection")

    st.write(
        "Enter the job posting information below. "
        "The system analyzes both textual and structured features."
    )

    st.divider()

    st.subheader("📝 Job Information")

    col1, col2 = st.columns(2)

    with col1:

        title = st.text_input(
            "Job Title",
            placeholder="Example: Software Developer"
        )

        location = st.text_input(
            "Location",
            placeholder="Example: New York, NY"
        )

        department = st.text_input(
            "Department",
            placeholder="Example: Engineering"
        )

    with col2:

        employment_type = st.selectbox(
            "Employment Type",
            [
                "Unknown",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other"
            ]
        )

        required_experience = st.selectbox(
            "Required Experience",
            [
                "Unknown",
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive"
            ]
        )

        required_education = st.selectbox(
            "Required Education",
            [
                "Unknown",
                "High School or equivalent",
                "Associate Degree",
                "Bachelor's Degree",
                "Master's Degree",
                "Doctorate",
                "Professional"
            ]
        )

    industry = st.text_input(
        "Industry",
        placeholder="Example: Information Technology"
    )

    function = st.text_input(
        "Job Function",
        placeholder="Example: Engineering"
    )

    st.divider()

    st.subheader("🏢 Company Information")

    company_profile = st.text_area(
        "Company Profile",
        height=150,
        placeholder="Enter information about the company..."
    )

    st.subheader("📄 Job Description")

    description = st.text_area(
        "Description",
        height=220,
        placeholder="Enter the complete job description..."
    )

    st.subheader("⚙️ Job Requirements")

    requirements = st.text_area(
        "Requirements",
        height=180,
        placeholder="Enter required skills, qualifications and experience..."
    )

    st.subheader("🎁 Benefits")

    benefits = st.text_area(
        "Benefits",
        height=150,
        placeholder="Enter salary, benefits, incentives and other information..."
    )

    st.divider()

    st.subheader("🔧 Additional Job Attributes")

    col1, col2, col3 = st.columns(3)

    with col1:

        telecommuting = st.checkbox(
            "Remote / Telecommuting"
        )

    with col2:

        has_company_logo = st.checkbox(
            "Company Logo Available"
        )

    with col3:

        has_questions = st.checkbox(
            "Screening Questions Available"
        )

    st.divider()

    if st.button(
        "🔎 Analyze Job Posting",
        use_container_width=True,
        type="primary"
    ):

        if not title:

            st.warning(
                "Please enter the job title."
            )

        elif not description:

            st.warning(
                "Please enter the job description."
            )

        else:

            with st.spinner(
                "Analyzing job posting..."
            ):

                result, risk_score, risk_level = predict_job(
                    title=title,
                    company_profile=company_profile,
                    description=description,
                    requirements=requirements,
                    benefits=benefits,
                    location=location if location else "Unknown",
                    department=department if department else "Unknown",
                    employment_type=employment_type,
                    required_experience=required_experience,
                    required_education=required_education,
                    industry=industry if industry else "Unknown",
                    function=function if function else "Unknown",
                    telecommuting=int(telecommuting),
                    has_company_logo=int(has_company_logo),
                    has_questions=int(has_questions)
                )

                save_prediction(
                    st.session_state.user_id,
                    title,
                    result,
                    risk_level,
                    float(risk_score)
                )

            st.divider()

            st.subheader("📊 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                if result == "Fraudulent":

                    st.error(
                        f"🚨 {result}"
                    )

                else:

                    st.success(
                        f"✅ {result}"
                    )

            with col2:

                if risk_level == "High Risk":

                    st.error(
                        f"🔴 {risk_level}"
                    )

                elif risk_level == "Medium Risk":

                    st.warning(
                        f"🟠 {risk_level}"
                    )

                elif risk_level == "Low Risk":

                    st.info(
                        f"🟡 {risk_level}"
                    )

                else:

                    st.success(
                        f"🟢 {risk_level}"
                    )

            with col3:

                st.metric(
                    "Fraud Risk Score",
                    f"{risk_score:.2f}%"
                )

            st.divider()

            st.subheader("Fraud Risk Level")

            st.progress(
                min(
                    max(
                        risk_score / 100,
                        0.0
                    ),
                    1.0
                )
            )

            if risk_score >= 75:

                st.error(
                    f"🔴 High Fraud Risk — {risk_score:.2f}%"
                )

            elif risk_score >= 50:

                st.warning(
                    f"🟠 Medium Fraud Risk — {risk_score:.2f}%"
                )

            elif risk_score >= 25:

                st.info(
                    f"🟡 Low Fraud Risk — {risk_score:.2f}%"
                )

            else:

                st.success(
                    f"🟢 Very Low Fraud Risk — {risk_score:.2f}%"
                )

            st.caption(
                "Risk score represents the calibrated model estimate "
                "of fraudulent-job risk and should not be treated as absolute proof."
            )

            st.divider()

            if result == "Fraudulent":

                st.error(
                    """
                    ### ⚠️ Potential Fraud Detected

                    The machine learning model identified patterns
                    associated with fraudulent job postings.

                    Before applying, verify:

                    • Company identity  
                    • Recruiter information  
                    • Salary claims  
                    • Contact details  
                    • Payment or registration requests  
                    • Interview process  
                    • Official company website
                    """
                )

            else:

                st.success(
                    """
                    ### ✅ Job Appears Genuine

                    The model did not identify strong fraudulent
                    patterns in this posting.

                    However, this prediction does not guarantee that
                    the job is completely legitimate.
                    """
                )


def history_page():

    st.title("📜 Prediction History")

    history = get_prediction_history(
        st.session_state.user_id
    )

    if not history:

        st.info(
            "No prediction history available."
        )

        return

    df = pd.DataFrame(
        history,
        columns=[
            "ID",
            "Job Title",
            "Prediction",
            "Risk Level",
            "Risk Score",
            "Date"
        ]
    )

    df["Risk Score"] = df["Risk Score"].apply(
        lambda x: f"{float(x):.2f}%"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    genuine_count = sum(
        1 for row in history
        if row[2] == "Genuine"
    )

    fraudulent_count = sum(
        1 for row in history
        if row[2] == "Fraudulent"
    )

    st.subheader("Prediction Distribution")

    chart_data = pd.DataFrame(
        {
            "Prediction": [
                "Genuine",
                "Fraudulent"
            ],
            "Count": [
                genuine_count,
                fraudulent_count
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Prediction")
    )


def analytics_page():

    st.title("📊 Model Analytics")

    st.write(
        "Performance comparison using text and structured job features."
    )

    metrics = pd.DataFrame(
        {
            "Model": [
                "Linear SVM",
                "Logistic Regression",
                "Random Forest",
                "Naive Bayes"
            ],
            "Accuracy": [
                99.19,
                97.90,
                98.32,
                96.95
            ],
            "Precision": [
                95.00,
                72.48,
                97.48,
                97.06
            ],
            "Recall": [
                87.86,
                91.33,
                67.05,
                38.15
            ],
            "F1 Score": [
                91.29,
                80.82,
                79.45,
                54.77
            ],
            "ROC-AUC": [
                99.12,
                99.05,
                99.09,
                95.96
            ]
        }
    )

    st.subheader("Model Comparison")

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Accuracy")

        st.bar_chart(
            metrics[
                ["Model", "Accuracy"]
            ].set_index("Model")
        )

    with col2:

        st.subheader("F1 Score")

        st.bar_chart(
            metrics[
                ["Model", "F1 Score"]
            ].set_index("Model")
        )

    st.divider()

    st.subheader("🏆 Selected Model")

    st.success(
        """
        Linear SVM achieved the strongest overall performance.

        Accuracy: 99.19%

        Precision: 95.00%

        Recall: 87.86%

        F1 Score: 91.29%

        ROC-AUC: 99.12%
        """
    )

    st.info(
        """
        The structured-feature model improved the F1 Score
        compared with the previous text-only Linear SVM.

        F1 Score increased from 89.43% to 91.29%.

        Accuracy increased from 99.02% to 99.19%.

        ROC-AUC was slightly lower than the previous text-only model.
        """
    )


def profile_page():

    st.title("👤 Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Personal Information")

        st.write(
            f"**Name:** {st.session_state.user_name}"
        )

        st.write(
            f"**Email:** {st.session_state.user_email}"
        )

    with col2:

        st.subheader("Account Information")

        st.write(
            "**Account Type:** Standard User"
        )

        st.write(
            "**System Access:** Active"
        )


def main_app():

    st.sidebar.title(
        "🛡️ Fake Job Detection"
    )

    st.sidebar.write(
        f"Welcome, **{st.session_state.user_name}**"
    )

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Detect Job",
            "📜 Prediction History",
            "📊 Analytics",
            "👤 Profile"
        ]
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.page = "login"

        for key in [
            "user_id",
            "user_name",
            "user_email"
        ]:

            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    if page == "🏠 Dashboard":

        dashboard_page()

    elif page == "🔍 Detect Job":

        detect_job_page()

    elif page == "📜 Prediction History":

        history_page()

    elif page == "📊 Analytics":

        analytics_page()

    elif page == "👤 Profile":

        profile_page()


if st.session_state.logged_in:

    main_app()

else:

    if st.session_state.page == "register":

        register_page()

    else:

        login_page()