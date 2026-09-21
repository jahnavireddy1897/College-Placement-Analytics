import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="College Placement Analytics",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("placement_model.pkl")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/placement_data.csv")

# ==========================================
# TITLE
# ==========================================

st.title("🎓 College Placement Analytics & Prediction")
st.write(
    "Analyze student placement data and predict placement outcomes using Machine Learning."
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Dashboard",
        "🎯 Placement Prediction",
        "📈 Model Evaluation",
        "📋 Student Data"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "📊 Dashboard":

    st.header("📊 Placement Analytics Dashboard")

    # --------------------------------------
    # KPIs
    # --------------------------------------

    total_students = len(df)

    placed_students = int(df["Placement"].sum())

    not_placed_students = total_students - placed_students

    placement_rate = (
        placed_students / total_students * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    col2.metric(
        "✅ Placed Students",
        placed_students
    )

    col3.metric(
        "❌ Not Placed",
        not_placed_students
    )

    col4.metric(
        "📈 Placement Rate",
        f"{placement_rate:.2f}%"
    )

    st.divider()

    # --------------------------------------
    # BRANCH FILTER
    # --------------------------------------

    branches = ["All"] + sorted(
        df["Branch"].unique().tolist()
    )

    selected_branch = st.selectbox(
        "Select Branch",
        branches
    )

    if selected_branch != "All":

        filtered_df = df[
            df["Branch"] == selected_branch
        ]

    else:

        filtered_df = df

    # --------------------------------------
    # BRANCH PLACEMENT ANALYSIS
    # --------------------------------------

    st.subheader("📌 Branch-wise Placement")

    branch_analysis = (
        filtered_df
        .groupby("Branch")["Placement"]
        .agg(
            Total_Students="count",
            Placed_Students="sum"
        )
        .reset_index()
    )

    branch_analysis["Placement_Rate"] = (
        branch_analysis["Placed_Students"]
        / branch_analysis["Total_Students"]
        * 100
    ).round(2)

    st.dataframe(
        branch_analysis,
        use_container_width=True
    )

    # --------------------------------------
    # CHARTS
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Placement by Branch")

        chart_data = (
            branch_analysis
            .set_index("Branch")[
                ["Placed_Students", "Total_Students"]
            ]
        )

        st.bar_chart(chart_data)

    with col2:

        st.subheader("📈 Average CGPA by Placement")

        cgpa_data = (
            filtered_df
            .groupby("Placement")["CGPA"]
            .mean()
        )

        cgpa_data.index = [
            "Not Placed" if x == 0 else "Placed"
            for x in cgpa_data.index
        ]

        st.bar_chart(cgpa_data)

    # --------------------------------------
    # PACKAGE ANALYSIS
    # --------------------------------------

    st.subheader("💰 Package Analysis")

    placed_df = filtered_df[
        filtered_df["Placement"] == 1
    ]

    if len(placed_df) > 0:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Package",
            f"{placed_df['Package_LPA'].mean():.2f} LPA"
        )

        col2.metric(
            "Highest Package",
            f"{placed_df['Package_LPA'].max():.2f} LPA"
        )

        col3.metric(
            "Lowest Package",
            f"{placed_df['Package_LPA'].min():.2f} LPA"
        )

# ==========================================
# PLACEMENT PREDICTION
# ==========================================

elif page == "🎯 Placement Prediction":

    st.header("🎯 Student Placement Prediction")

    st.write(
        "Enter student details below to predict the placement outcome."
    )

    # --------------------------------------
    # PERSONAL / ACADEMIC DETAILS
    # --------------------------------------

    st.subheader("📚 Academic Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.01
        )

    with col2:

        tenth = st.number_input(
            "10th Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.1
        )

    with col3:

        twelfth = st.number_input(
            "12th Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.1
        )

    col1, col2 = st.columns(2)

    with col1:

        backlogs = st.number_input(
            "Number of Backlogs",
            min_value=0,
            max_value=20,
            value=0
        )

    with col2:

        internships = st.number_input(
            "Number of Internships",
            min_value=0,
            max_value=10,
            value=1
        )

    # --------------------------------------
    # PROJECTS / CERTIFICATIONS
    # --------------------------------------

    st.subheader("💻 Experience & Skills")

    col1, col2 = st.columns(2)

    with col1:

        projects = st.number_input(
            "Number of Projects",
            min_value=0,
            max_value=20,
            value=2
        )

    with col2:

        certifications = st.number_input(
            "Number of Certifications",
            min_value=0,
            max_value=20,
            value=2
        )

    # --------------------------------------
    # SCORES
    # --------------------------------------

    st.subheader("📝 Assessment Scores")

    col1, col2, col3 = st.columns(3)

    with col1:

        aptitude = st.number_input(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70
        )

    with col2:

        communication = st.number_input(
            "Communication Score",
            min_value=0,
            max_value=100,
            value=70
        )

    with col3:

        dsa = st.number_input(
            "DSA Score",
            min_value=0,
            max_value=100,
            value=65
        )

    # --------------------------------------
    # TECHNICAL SKILLS
    # --------------------------------------

    st.subheader("🛠️ Technical Skills")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        python_skill = st.checkbox("Python")

        sql_skill = st.checkbox("SQL")

    with col2:

        java_skill = st.checkbox("Java")

        ml_skill = st.checkbox("Machine Learning")

    with col3:

        web_skill = st.checkbox("Web Development")

        excel_skill = st.checkbox("Excel")

    # --------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------

    st.divider()

    if st.button(
        "🚀 Predict Placement",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                cgpa,
                tenth,
                twelfth,
                backlogs,
                internships,
                projects,
                certifications,
                aptitude,
                communication,
                dsa,
                int(python_skill),
                int(sql_skill),
                int(java_skill),
                int(ml_skill),
                int(web_skill),
                int(excel_skill)
            ]],
            columns=[
                "CGPA",
                "Tenth_Percentage",
                "Twelfth_Percentage",
                "Backlogs",
                "Internships",
                "Projects",
                "Certifications",
                "Aptitude_Score",
                "Communication_Score",
                "DSA_Score",
                "Python",
                "SQL",
                "Java",
                "Machine_Learning",
                "Web_Development",
                "Excel"
            ]
        )

        # Prediction

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0]

        placement_probability = (
            probability[1] * 100
        )

        # ----------------------------------
        # RESULT
        # ----------------------------------

        st.divider()

        if prediction == 1:

            st.success(
                "🎉 Placement Prediction: PLACED"
            )

        else:

            st.error(
                "⚠️ Placement Prediction: NOT PLACED"
            )

        st.metric(
            "📊 Placement Probability",
            f"{placement_probability:.2f}%"
        )

        # ----------------------------------
        # PROGRESS BAR
        # ----------------------------------

        st.progress(
            int(placement_probability)
        )

        # ----------------------------------
        # INTERPRETATION
        # ----------------------------------

        if placement_probability >= 75:

            st.info(
                "The model predicts a relatively high placement probability."
            )

        elif placement_probability >= 50:

            st.warning(
                "The model predicts a moderate placement probability."
            )

        else:

            st.warning(
                "The model predicts a lower placement probability. "
                "Improving academic performance, technical skills, "
                "projects, aptitude and communication may help."
            )

# ==========================================
# MODEL EVALUATION
# ==========================================

elif page == "📈 Model Evaluation":

    st.header("📈 Machine Learning Model Evaluation")

    st.write(
        "Comparison and evaluation of machine learning models "
        "for student placement prediction."
    )

    # --------------------------------------
    # LOAD MODEL COMPARISON
    # --------------------------------------

    evaluation_df = pd.read_csv(
        "model_comparison.csv"
    )

    # --------------------------------------
    # KPI METRICS
    # --------------------------------------

    best_row = evaluation_df[
        evaluation_df["Model"] == "Logistic Regression"
    ].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{best_row['Accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{best_row['Precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{best_row['Recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{best_row['F1_Score'] * 100:.2f}%"
    )

    col5.metric(
        "ROC-AUC",
        f"{best_row['ROC_AUC']:.3f}"
    )

    st.divider()

    # --------------------------------------
    # MODEL COMPARISON TABLE
    # --------------------------------------

    st.subheader("🤖 Model Comparison")

    display_df = evaluation_df.copy()

    display_df["Accuracy"] = (
        display_df["Accuracy"] * 100
    ).round(2)

    display_df["Precision"] = (
        display_df["Precision"] * 100
    ).round(2)

    display_df["Recall"] = (
        display_df["Recall"] * 100
    ).round(2)

    display_df["F1_Score"] = (
        display_df["F1_Score"] * 100
    ).round(2)

    display_df["ROC_AUC"] = (
        display_df["ROC_AUC"]
    ).round(3)

    display_df["CV_Accuracy"] = (
        display_df["CV_Accuracy"] * 100
    ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # --------------------------------------
    # MODEL COMPARISON CHART
    # --------------------------------------

    st.subheader("📊 Model Performance Comparison")

    chart_df = evaluation_df.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score"
        ]
    ] * 100

    st.bar_chart(
        chart_df
    )

    # --------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------

    st.subheader("🔢 Confusion Matrix")

    try:

        st.image(
            "confusion_matrix.png",
            caption="Logistic Regression Confusion Matrix",
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "confusion_matrix.png was not found."
        )

    # --------------------------------------
    # INTERPRETATION
    # --------------------------------------

    st.subheader("📝 Model Interpretation")

    st.info(
        """
        The evaluation compares Logistic Regression, Random Forest,
        and Gradient Boosting.

        Logistic Regression was selected based on the highest F1-score
        among the evaluated models.

        The model achieved 70.50% accuracy, 68.55% precision,
        80.95% recall, 74.24% F1-score and 0.803 ROC-AUC
        on the test set.

        These results describe performance on this dataset and
        should not be interpreted as a guarantee of an individual
        student's placement outcome.
        """
    )
# ==========================================
# STUDENT DATA
# ==========================================

elif page == "📋 Student Data":

    st.header("📋 Student Profile")

    st.write(
        f"Showing {len(df)} student records."
    )

    # --------------------------------------------
    # SELECT STUDENT
    # --------------------------------------------

    student_options = df["Student_ID"].tolist()

    selected_id = st.selectbox(
        "🔎 Select Student",
        student_options,
        format_func=lambda x: (
            f"{x} - "
            f"{df.loc[df['Student_ID'] == x, 'Student_Name'].iloc[0]}"
            if not pd.isna(
                df.loc[df["Student_ID"] == x, "Student_Name"].iloc[0]
            )
            and str(
                df.loc[df["Student_ID"] == x, "Student_Name"].iloc[0]
            ).strip() != ""
            else f"{x} - Student Name Not Added"
        )
    )

    student = df[df["Student_ID"] == selected_id].iloc[0]

    # --------------------------------------------
    # STUDENT INFORMATION
    # --------------------------------------------

    st.subheader("👤 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**🆔 Student ID:**", student["Student_ID"])

        name = student["Student_Name"]
        if pd.isna(name) or str(name).strip() == "":
            name = "Not Provided"

        st.write("**👤 Student Name:**", name)

        batch = student["Admitted_Batch"]
        if pd.isna(batch) or str(batch).strip() == "":
            batch = "Not Provided"

        st.write("**🎓 Admitted Batch:**", batch)

        st.write("**🏫 Branch:**", student["Branch"])

        st.write("**📈 CGPA:**", student["CGPA"])

    with col2:

        email = student["Email"]
        if pd.isna(email) or str(email).strip() == "":
            email = "Not Provided"

        st.write("**📧 Email:**", email)

        mobile = student["Mobile"]
        if pd.isna(mobile) or str(mobile).strip() == "":
            mobile = "Not Provided"

        st.write("**📱 Mobile:**", mobile)

        linkedin = student["LinkedIn"]
        if pd.isna(linkedin) or str(linkedin).strip() == "":
            st.write("**🔗 LinkedIn:** Not Provided")
        else:
            st.markdown(
                f"**🔗 LinkedIn:** [View Profile]({linkedin})"
            )

        github = student["GitHub"]
        if pd.isna(github) or str(github).strip() == "":
            st.write("**💻 GitHub:** Not Provided")
        else:
            st.markdown(
                f"**💻 GitHub:** [View Profile]({github})"
            )

    # --------------------------------------------
    # PLACEMENT INFORMATION
    # --------------------------------------------

    st.subheader("📊 Placement Information")

    col3, col4 = st.columns(2)

    with col3:
        st.write(
            "**Placement Status:**",
            student["Placement"]
        )

    with col4:
        st.write(
            "**Package (LPA):**",
            student["Package_LPA"]
        )

    # --------------------------------------------
    # ACADEMIC & SKILLS INFORMATION
    # --------------------------------------------

    st.subheader("📚 Academic & Skills Information")

    academic_data = pd.DataFrame({
        "Details": [
            "10th Percentage",
            "12th Percentage",
            "Backlogs",
            "Internships",
            "Projects",
            "Certifications",
            "Aptitude Score",
            "Communication Score",
            "DSA Score",
            "Python",
            "SQL",
            "Java",
            "Machine Learning",
            "Web Development",
            "Excel"
        ],
        "Value": [
            student["Tenth_Percentage"],
            student["Twelfth_Percentage"],
            student["Backlogs"],
            student["Internships"],
            student["Projects"],
            student["Certifications"],
            student["Aptitude_Score"],
            student["Communication_Score"],
            student["DSA_Score"],
            student["Python"],
            student["SQL"],
            student["Java"],
            student["Machine_Learning"],
            student["Web_Development"],
            student["Excel"]
        ]
    })

    st.dataframe(
        academic_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------
    # COMPLETE DATA
    # --------------------------------------------

    with st.expander("🔍 View Complete Student Record"):

        st.dataframe(
            student.to_frame("Value"),
            use_container_width=True
        )

    # --------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------

    st.subheader("📥 Download Dataset")

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Placement Dataset",
        data=csv,
        file_name="placement_data.csv",
        mime="text/csv"
    )