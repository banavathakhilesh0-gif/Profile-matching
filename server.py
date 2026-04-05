from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import csv
import datetime
import os
from scoring_engine import ScoringEngine
from feedback_loop import AdaptiveFeedbackLoop

app = FastAPI(title="Profile Matcher AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, 'users.csv')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'feedback.csv')
CACHE_FILE = os.path.join(BASE_DIR, 'pipeline_cache.pkl')

# Load engine once on startup
engine = ScoringEngine()
loop = AdaptiveFeedbackLoop(engine)

def reload_engine():
    global engine, loop
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    engine = ScoringEngine()
    loop = AdaptiveFeedbackLoop(engine)

# Serve frontend
STATIC_DIR = os.path.join(BASE_DIR, 'static')
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Profile Matcher API"}

@app.get("/api/users")
def get_users():
    users = []
    for uid, row in engine.users_df.iterrows():
        users.append({
            'user_id': uid,
            'name': str(row['name']),
            'age': int(row['age']),
            'location': str(row['location']),
            'profession': str(row['profession']),
            'experience_years': int(row['experience_years']),
            'mbti': str(row['mbti']),
            'interests': str(row['interests']),
            'professional_summary': str(row['professional_summary']),
            'about_me': str(row['about_me']),
        })
    return users

@app.get("/api/matches/{user_id}")
def get_matches(user_id: str, limit: int = 5):
    try:
        if os.path.exists(FEEDBACK_FILE):
            engine.pipeline.feedback_df = pd.read_csv(FEEDBACK_FILE)
            loop.feedback_df = engine.pipeline.feedback_df
    except Exception as e:
        print(f"Error loading feedback: {e}")
        engine.pipeline.feedback_df = pd.DataFrame(columns=['user_id', 'matched_user_id', 'action', 'timestamp'])

    weights = loop.get_optimal_weights(user_id)
    all_users = engine.users_df.index.tolist()
    
    # Get list of user IDs that have already received feedback from this user
    interacted_users = engine.pipeline.feedback_df[engine.pipeline.feedback_df['user_id'] == user_id]['matched_user_id'].tolist() if not engine.pipeline.feedback_df.empty else []
    
    all_scores = []

    for other_user in all_users:
        # Exclude self and already interacted users
        if other_user == user_id or other_user in interacted_users:
            continue
        try:
            score_data = engine.get_score(user_id, other_user, weights=weights)
            match_info = engine.users_df.loc[other_user]
            all_scores.append({
                'user_id': other_user,
                'name': str(match_info['name']),
                'profession': str(match_info['profession']),
                'location': str(match_info['location']),
                'mbti': str(match_info['mbti']),
                'interests': str(match_info['interests']),
                'professional_summary': str(match_info['professional_summary']),
                'about_me': str(match_info['about_me']),
                'total_score': float(score_data['total_score']),
                'text_sim': float(score_data['text_sim']),
                'mbti_match': float(score_data['mbti_match']),
                'location_match': float(score_data['location_match']),
            })
        except Exception as e:
            print(f"Error scoring user {other_user}: {e}")
            continue

    all_scores.sort(key=lambda x: x['total_score'], reverse=True)
    return {
        'weights': weights,
        'matches': all_scores[:limit],
        'total': len(all_scores)
    }

class FeedbackPayload(BaseModel):
    user_id: str
    matched_user_id: str
    action: int

@app.post("/api/feedback")
def record_feedback(payload: FeedbackPayload):
    now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    with open(FEEDBACK_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([payload.user_id, payload.matched_user_id, payload.action, now])
    return {"status": "ok"}

class NewUserPayload(BaseModel):
    name: str
    age: int
    location: str
    profession: str
    experience_years: int
    professional_summary: str
    about_me: str
    mbti: str
    interests: str

@app.post("/api/users")
def create_user(payload: NewUserPayload):
    users_df_raw = pd.read_csv(USERS_FILE)
    last_id = users_df_raw['user_id'].iloc[-1]
    last_num = int(last_id[1:])
    new_id = f"U{last_num + 1:03d}"
    new_row = {
        'user_id': new_id,
        'name': payload.name,
        'age': payload.age,
        'location': payload.location,
        'profession': payload.profession,
        'experience_years': payload.experience_years,
        'professional_summary': payload.professional_summary,
        'about_me': payload.about_me,
        'mbti': payload.mbti,
        'interests': payload.interests,
    }
    with open(USERS_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_row.keys())
        writer.writerow(new_row)
    reload_engine()
    return {"status": "ok", "user_id": new_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
