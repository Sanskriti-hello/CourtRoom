# streamlit_app.py

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.judge_agent import JudgeAgent
from agents.lawyer_agent import LawyerAgent
from database.courtroom_db import CourtroomDB
from run_trial import init_agents, run_trial


# Streamlit app title
st.title("Courtroom Trial Simulator")

# Upload case details and past cases
st.header("Step 1: Upload Case Details")
case_details = st.text_area("Enter the details of the case (long paragraph)", height=200)

st.header("Step 2: Upload Past Case Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file with past cases", type=["csv"])

past_cases_df = pd.DataFrame() 
past_cases = ""

if uploaded_file is not None:
    past_cases_df = pd.read_csv(uploaded_file)
    # Assuming the CSV has a column named 'text' for case descriptions
    past_cases = "\n".join(past_cases_df['text'].head(5).tolist())  # Take the top 5 cases for context
else:
    past_cases = ""

# Step 3: Set the number of rounds for the trial
rounds = st.slider("Select number of rounds for arguments", min_value=1, max_value=5, value=2)

# Step 4: Run the trial when the user presses the button
if st.button("Start Trial"):
    if not case_details:
        st.error("Please enter the case details before starting the trial.")
    else:
        # Use fallback if no valid file uploaded
        if uploaded_file is None or "text" not in past_cases_df.columns:
            st.warning("No valid past case data uploaded. The trial will proceed without it.")
            db = CourtroomDB(pd.DataFrame())  # fallback: empty DataFrame
            past_cases = ""
        else:
            db = CourtroomDB(past_cases_df)

        # Initialize agents
        defense, prosecution, defendant, plaintiff, judge = init_agents(db)

        # Run the trial
        result = run_trial(plaintiff, prosecution, defense, defendant, judge, case_details, past_cases, rounds)

        # Display results
        st.header("Trial Results")

        st.subheader("Case Background")
        st.write(result["case"])

        st.subheader("Trial History")
        for entry in result["history"]:
            st.write(f"{entry['role'].upper()} ({entry['name']}): {entry['content']}")

        st.subheader("Judge's Reflections")
        st.write(result["reflections"])

        st.subheader("Judge's Verdict")
        st.write(result["verdict"])

# Display app instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter the details of the case in the provided text area.
2. Upload a CSV file containing past cases, where the 'text' column should contain case descriptions.
3. Set the number of rounds you want the trial to go through.
4. Press "Start Trial" to begin the trial.
5. The results will show the opening statements, arguments, rebuttals, and the final verdict.
""")
