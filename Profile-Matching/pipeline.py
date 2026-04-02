import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class DataPipeline:
    def __init__(self, users_file='users.csv', feedback_file='feedback.csv'):
        self.users_file = users_file
        self.feedback_file = feedback_file
        self.users_df = None
        self.feedback_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.user_index_map = {}

    def load_data(self):
        print("Loading data...")
        self.users_df = pd.read_csv(self.users_file)
        # Ensure user_id is the index for fast lookup
        self.users_df.set_index('user_id', drop=False, inplace=True)
        
        # Create map from user_id to row index for matrix lookups
        for idx, row in self.users_df.reset_index(drop=True).iterrows():
            self.user_index_map[row['user_id']] = idx
        
        try:
            self.feedback_df = pd.read_csv(self.feedback_file)
            print(f"Loaded {len(self.users_df)} users and {len(self.feedback_df)} feedbacks.")
        except Exception as e:
            print(f"Warning: Could not load feedback data. {e}")

    def prepare_text_features(self):
        """Combines text fields and computes TF-IDF embeddings."""
        print("Preparing text features...")
        # Combine text from professional_summary, about_me, and interests
        self.users_df['combined_text'] = (
            self.users_df['profession'].fillna('') + " " +
            self.users_df['professional_summary'].fillna('') + " " + 
            self.users_df['about_me'].fillna('') + " " + 
            self.users_df['interests'].fillna('')
        )
        
        # Compute TF-IDF Matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.users_df['combined_text'])
        print(f"TF-IDF Matrix shape: {self.tfidf_matrix.shape}")

    def save_pipeline(self, cache_file='pipeline_cache.pkl'):
        """Saves the fitted vectorizer and tfidf matrix to avoid recompiling."""
        print(f"Saving pipeline to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'user_index_map': self.user_index_map
            }, f)
            
    def load_pipeline(self, cache_file='pipeline_cache.pkl'):
        if os.path.exists(cache_file):
            print(f"Loading cached pipeline from {cache_file}...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.tfidf_matrix = data['tfidf_matrix']
                self.user_index_map = data['user_index_map']
            return True
        return False

    def run(self):
        self.load_data()
        self.prepare_text_features()
        self.save_pipeline()
        print("Data Pipeline Execution Complete.\n")

if __name__ == "__main__":
    pipeline = DataPipeline('users.csv', 'feedback.csv')
    pipeline.run()
