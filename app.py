# app.py

import streamlit as st
import os
from crewai import Agent, Task, Crew
from langchain_community.llms import OpenAI
from utils import get_openai_api_key

# Set API key
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# --- Define Agents ---
profile_analyzer = Agent(
    role="User Profile Analyzer",
    goal="Analyze the user's health and lifestyle data to identify key factors that should influence their nutritional planning.",
    backstory="You are a data-driven analyst specializing in health and nutrition personalization. You interpret user input — such as age, gender, weight, height, medical conditions, allergies, and preferences — and extract insights that will guide the Nutrition Assistant.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

meal_planner = Agent(
    role="Meal Planner",
    goal="Design a personalized weekly meal plan aligned with the user's dietary preferences, goals, and health requirements.",
    backstory="You are an expert in nutrition and culinary planning. You use profile data to create well-balanced, practical, and diverse meals.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

health_coach = Agent(
    role="Health Coach",
    goal="Motivate the user and provide actionable wellness tips aligned with their nutrition and health goals.",
    backstory="You are a warm, supportive virtual coach dedicated to helping users stay on track with their health journey using behavioral science tips.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# --- Streamlit UI ---
st.title("🥗 Smart AI-Powered Nutrition Planning Assistant (SANA)")

st.markdown("Enter your personal details to receive a personalized weekly meal plan and wellness guidance.")

user_profile = st.text_area(
    "📝 Describe your profile (e.g., age, weight, health goals, preferences, allergies)",
    "28-year-old vegetarian female, 160cm, 65kg, goal: weight loss, high activity, prefers Indian cuisine"
)

if st.button("Generate Plan"):
    with st.spinner("Processing your profile..."):

        # --- Define Tasks using formatted input ---
        profile_analysis_task = Task(
            description=(
                f"Analyze the following user profile data: {user_profile}. "
                "Identify key health risks, preferences, or lifestyle factors that impact nutrition planning. "
                "Summarize important contraindications or special considerations."
            ),
            expected_output="A detailed analysis report highlighting important health and lifestyle factors.",
            agent=profile_analyzer,
        )

        meal_planning_task = Task(
            description=(
                f"Using the user's profile: {user_profile}, create a balanced weekly meal plan. "
                "Ensure meals match preferences, allergies, and nutritional goals. "
                "Include breakfast, lunch, dinner, and snack ideas with estimated calories."
            ),
            expected_output="A weekly meal plan formatted by day and time, with nutritional highlights.",
            agent=meal_planner,
        )

        health_coach_task = Task(
            description=(
                f"Review the user profile: {user_profile}. "
                "Provide motivational advice and habit-forming tips tailored to the user. "
                "Use an encouraging and supportive tone."
            ),
            expected_output="A personalized health coaching message with actionable wellness tips.",
            agent=health_coach,
        )

        # --- Create and Run Crew ---
        crew = Crew(
            agents=[profile_analyzer, meal_planner, health_coach],
            tasks=[profile_analysis_task, meal_planning_task, health_coach_task],
            verbose=2
        )

        result = crew.kickoff()  # No arguments passed here!

    st.success("✅ Plan Generated!")
    st.markdown("### 📋 Output")
    st.markdown(result)
