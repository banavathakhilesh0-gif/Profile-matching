# Profile Matcher AI 🚀
An advanced, hybrid recommendation system designed to match professionals based on their technical skills, personality types (MBTI), and geographical location. The system employs an **Adaptive Machine Learning Feedback Loop** to refine its matching weights based on user interactions.

## 🏗️ Core Architecture

### 1. Hybrid Scoring Engine
The system calculates a compatibility score between two users using three distinct dimensions:
- **NLP Text Similarity (40-60% base weight)**: Uses TF-IDF vectorization and Cosine Similarity to compare "Professional Summaries" and "About Me" sections.
- **MBTI Synergy Logic (20-40% base weight)**: A custom compatibility matrix that scores personality type pairings based on established psychological synergy patterns.
- **Location Matching (10-20% base weight)**: Prioritizes local matches while allowing for global professional connections.

### 2. Adaptive ML Feedback Loop
Instead of static weights, the system evolves using **Logistic Regression**. Each time a user clicks "Accept" or "Reject":
1. The interaction is logged to `feedback.csv`.
2. The ML model analyzes the user's history and adjusts the weights ($w_1, w_2, w_3$) to match their individual preferences (e.g., if a user consistently accepts MBTI-synergistic profiles over professional ones, the MBTI weight increases automatically).

## ✨ Premium UI Features
- **Modern Search & Autocomplete**: Find yourself in a 150+ user database instantly.
- **Glassmorphism Design**: High-end dark mode aesthetic with vibrant gradients and animated blobs.
- **Dynamic Scoring Rings**: Visual representation of match percentages using animated SVG ring gauges.
- **Clickable Match Cards**: Summarized initial view with full-detail expansion via interactive modals.
- **Real-time Feedback**: Instant system adaptation upon profile interaction.

## 📸 Project Screenshots

### 1. The Matching Experience
![Matching Feed](file:///C:/Users/rashm/.gemini/antigravity/brain/a3d67a5a-9c61-413b-a599-6a73ea120ffa/detailed_modal_view_1775024575820.png)
*Initial view showing top matches with their compatibility breakdowns.*

### 2. Detailed Profile Deep-Dive
![Profile Detail Modal](file:///C:/Users/rashm/.gemini/antigravity/brain/a3d67a5a-9c61-413b-a599-6a73ea120ffa/opened_modal_details_1775024594145.png)
*Clickable card expansion showing full professional summaries and interests.*

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Dependencies: `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `numpy`

### Installation
```bash
python -m pip install fastapi uvicorn pandas scikit-learn numpy
```

### Running the App
```bash
python server.py
```
Visit `http://127.0.0.1:8000` in your browser.

## 📁 File Structure
- `server.py`: FastAPI backend and REST endpoints.
- `static/index.html`: Premium Vanilla JS/CSS frontend.
- `scoring_engine.py`: The logic layer for NLP and MBTI scores.
- `feedback_loop.py`: The ML layer for weight adaptation.
- `pipeline.py`: Data ingestion and TF-IDF feature engineering.
- `users.csv`: Dataset containing 150 user profiles.
- `feedback.csv`: History of user interactions.

## 📄 License
MIT

## 👤 Author
Akhilesh
