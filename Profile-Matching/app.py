import streamlit as st
import pandas as pd
import datetime
import csv
import os
import pickle
from scoring_engine import ScoringEngine
from feedback_loop import AdaptiveFeedbackLoop

# Page Config
st.set_page_config(page_title="Profile Matcher AI", layout="wide")

USERS_FILE = 'users.csv'
FEEDBACK_FILE = 'feedback.csv'
CACHE_FILE = 'pipeline_cache.pkl'

MBTI_OPTIONS = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
                "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]

@st.cache_resource
def load_models():
    engine = ScoringEngine()
    loop = AdaptiveFeedbackLoop(engine)
    return engine, loop

engine, loop = load_models()

st.title("🤝 Intelligent Profile Matcher")
st.markdown("Find your best professional/personal matches using NLP Text Similarity, MBTI Logic, and Adaptive ML weights.")

# ─────────────────────────────────────
# SIDEBAR - Create New User FORM
# ─────────────────────────────────────
st.sidebar.markdown("## ➕ Register as a New User")

with st.sidebar.form("new_user_form", clear_on_submit=True):
    new_name        = st.text_input("Full Name *")
    new_age         = st.number_input("Age *", min_value=18, max_value=60, value=25)
    new_location    = st.text_input("City *")
    new_profession  = st.text_input("Profession *")
    new_exp         = st.number_input("Years of Experience *", min_value=0, max_value=40, value=1)
    new_prof_sum    = st.text_area("Professional Summary * (2-4 lines about your skills & goals)")
    new_about       = st.text_area("About Me * (2-4 lines about your personality & interests)")
    new_mbti        = st.selectbox("MBTI Type *", MBTI_OPTIONS)
    new_interests   = st.text_input("Interests (comma-separated, e.g. Chess, Reading, Hiking)")
    submitted       = st.form_submit_button("Create Profile 🚀")

    if submitted:
        if not all([new_name, new_location, new_profession, new_prof_sum, new_about, new_interests]):
            st.error("Please fill in all required fields marked with *")
        else:
            # Auto-generate next user_id
            users_df_raw = pd.read_csv(USERS_FILE)
            last_id = users_df_raw['user_id'].iloc[-1]  # e.g., "U150"
            last_num = int(last_id[1:])
            new_id = f"U{last_num + 1:03d}"

            new_row = {
                'user_id': new_id,
                'name': new_name,
                'age': int(new_age),
                'location': new_location,
                'profession': new_profession,
                'experience_years': int(new_exp),
                'professional_summary': new_prof_sum,
                'about_me': new_about,
                'mbti': new_mbti,
                'interests': new_interests
            }

            # Append to users.csv
            with open(USERS_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=new_row.keys())
                writer.writerow(new_row)

            # Delete pipeline cache so it rebuilds with new user
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)

            st.success(f"✅ Profile created! Your User ID is **{new_id}**. Please reload the page to see yourself in the dropdown.")
            st.cache_resource.clear()

st.sidebar.markdown("---")

# ─────────────────────────────────────
# SIDEBAR - User Selection
# ─────────────────────────────────────
users = engine.users_df.index.tolist()
selected_user = st.sidebar.selectbox("🔍 Select Your Profile (user_id):", users)

if selected_user:
    st.sidebar.subheader("Your Profile Overview")
    user_info = engine.users_df.loc[selected_user]
    st.sidebar.write(f"**Name:** {user_info['name']}")
    st.sidebar.write(f"**Profession:** {user_info['profession']} ({user_info['experience_years']} yrs)")
    st.sidebar.write(f"**MBTI:** {user_info['mbti']}")
    st.sidebar.write(f"**Location:** {user_info['location']}")

    # Reload feedback data fresh
    try:
        engine.pipeline.feedback_df = pd.read_csv(FEEDBACK_FILE)
        loop.feedback_df = engine.pipeline.feedback_df
    except:
        pass

    weights = loop.get_optimal_weights(selected_user)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 ML Personalized Weights")
    st.sidebar.write(f"NLP Text Similarity (w1): **{weights['w1']:.2%}**")
    st.sidebar.write(f"MBTI Compatibility (w2):  **{weights['w2']:.2%}**")
    st.sidebar.write(f"Location Match (w3):      **{weights['w3']:.2%}**")

    # ─────────────────────────────────────
    # MAIN AREA - Top 5 Matches
    # ─────────────────────────────────────
    all_scores = []
    for other_user in users:
        if other_user == selected_user:
            continue
        try:
            score_data = engine.get_score(selected_user, other_user, weights=weights)
            all_scores.append(score_data)
        except:
            continue

    all_scores.sort(key=lambda x: x['total_score'], reverse=True)
    top_5 = all_scores[:5]

    st.subheader("🌟 Your Top 5 Recommended Matches")

    def record_feedback(viewer_id, m_id, action):
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
        with open(FEEDBACK_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([viewer_id, m_id, action, now])
        st.toast(f"Feedback recorded for {m_id}!")

    for rank, match_data in enumerate(top_5, start=1):
        match_id = match_data['user_b']
        match_info = engine.users_df.loc[match_id]

        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### #{rank}: {match_info['name']} ({match_id})")
                st.write(f"**{match_info['profession']}** in {match_info['location']} | **MBTI:** {match_info['mbti']}")
                st.write(f"_{match_info['professional_summary']}_")
                st.caption(f"**Interests:** {match_info['interests']}")
            with col2:
                st.metric("Compatibility", f"{match_data['total_score']:.1%}")
                st.caption(f"Text: {match_data['text_sim']:.1%} | MBTI: {match_data['mbti_match']:.1%} | Loc: {match_data['location_match']}")

                c_a, c_r = st.columns(2)
                if c_a.button("👍", key=f"acc_{match_id}_{rank}"):
                    record_feedback(selected_user, match_id, 1)
                    st.rerun()
                if c_r.button("👎", key=f"rej_{match_id}_{rank}"):
                    record_feedback(selected_user, match_id, 0)
                    st.rerun()
            st.markdown("---")
