import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import DataPipeline

class ScoringEngine:
    def __init__(self, data_pipeline=None):
        if data_pipeline is None:
            self.pipeline = DataPipeline('users.csv', 'feedback.csv')
            # Load pipeline or run it if not cached
            if not self.pipeline.load_pipeline():
                self.pipeline.run()
        else:
            self.pipeline = data_pipeline
            
        self.users_df = self.pipeline.users_df
        if self.users_df is None:
            self.pipeline.load_data()
            self.users_df = self.pipeline.users_df

    def calculate_mbti_score(self, mbti1, mbti2):
        # A simple MBTI scoring logic: 
        # For this scope, exact match = 1.0, and subtract 0.25 for every differing letter.
        # As per PDF example: "INTJ and ENFP = 100" (which implies opposite intuition but ideal pairs exist).
        # We will use a standard simple synergy rule:
        # Ideal pairs (just a few examples from MBTI theory):
        ideal_pairs = [
            {'INTJ', 'ENFP'}, {'ENTJ', 'INFP'}, {'INFJ', 'ENTP'}, {'ENFJ', 'INTP'},
            {'ISTJ', 'ESTP'}, {'ISFJ', 'ESFP'}, {'ESTJ', 'ISTP'}, {'ESFJ', 'ISFP'}
        ]
        if {mbti1, mbti2} in ideal_pairs:
            return 1.0
            
        if mbti1 == mbti2:
            return 0.90 # high but maybe ideal pairs are better
            
        # Fallback to shared letters
        shared_letters = sum(1 for a, b in zip(mbti1, mbti2) if a == b)
        return shared_letters / 4.0

    def get_score(self, user_id_a, user_id_b, weights=None):
        """
        weights: dict with w1 (TextSim), w2 (MBTI), w3 (Location).
        Defaults to equal weights if None.
        """
        if weights is None:
            weights = {'w1': 1/3, 'w2': 1/3, 'w3': 1/3}
            
        if user_id_a not in self.users_df.index or user_id_b not in self.users_df.index:
            raise ValueError("Invalid user_id provided.")

        user_a = self.users_df.loc[user_id_a]
        user_b = self.users_df.loc[user_id_b]

        # 1. Location match
        loc_score = 1.0 if user_a['location'] == user_b['location'] else 0.0

        # 2. MBTI match
        mbti_score = self.calculate_mbti_score(user_a['mbti'], user_b['mbti'])

        # 3. Text Similarity (NLP)
        idx_a = self.pipeline.user_index_map[user_id_a]
        idx_b = self.pipeline.user_index_map[user_id_b]
        
        vec_a = self.pipeline.tfidf_matrix[idx_a]
        vec_b = self.pipeline.tfidf_matrix[idx_b]
        
        text_score = cosine_similarity(vec_a, vec_b)[0][0]

        # Ensure text score is strictly between 0 and 1
        text_score = max(0.0, min(1.0, text_score))

        # 4. Total Weighted Score
        total_score = (weights['w1'] * text_score) + (weights['w2'] * mbti_score) + (weights['w3'] * loc_score)

        return {
            'user_a': user_id_a,
            'user_b': user_id_b,
            'text_sim': text_score,
            'mbti_match': mbti_score,
            'location_match': loc_score,
            'total_score': total_score,
            'weights_used': weights
        }

if __name__ == "__main__":
    engine = ScoringEngine()
    score = engine.get_score('U001', 'U002')
    print("Match Score U001 vs U002:")
    print(score)
