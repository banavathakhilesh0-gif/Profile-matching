import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scoring_engine import ScoringEngine

class AdaptiveFeedbackLoop:
    def __init__(self, engine=None):
        if engine is None:
            self.engine = ScoringEngine()
        else:
            self.engine = engine
            
        self.feedback_df = self.engine.pipeline.feedback_df

    def get_optimal_weights(self, user_id):
        # Filter feedback for this user where they are the 'actor' (user_id viewer)
        # Assuming the first 'user_id' column is the viewer.
        user_feedback = self.feedback_df[self.feedback_df['user_id'] == user_id]
        
        default_weights = {'w1': 1/3, 'w2': 1/3, 'w3': 1/3}
        
        # If no feedback or not enough varied feedback, return default
        if len(user_feedback) < 3:
            return default_weights
            
        # Check if there are both positive and negative samples
        # Logistic Regression needs at least one of each class to train properly
        unique_classes = user_feedback['action'].unique()
        if len(unique_classes) < 2:
            return default_weights
            
        X = []
        y = []
        
        # Gather sub-scores for each interaction
        for idx, row in user_feedback.iterrows():
            matched_id = row['matched_user_id']
            action = int(row['action'])
            
            # Fetch raw scores using default equal weights (the components are what we care about)
            score_data = self.engine.get_score(user_id, matched_id, weights=default_weights)
            
            features = [
                score_data['text_sim'],
                score_data['mbti_match'],
                score_data['location_match']
            ]
            X.append(features)
            y.append(action)
            
        X = np.array(X)
        y = np.array(y)
        
        # Train Logistic Regression
        model = LogisticRegression(class_weight='balanced')
        model.fit(X, y)
        
        # Extract coefficients. 
        # Logistic Regression coeff implies importance of the feature.
        # We take the absolutes or clip to positive to form our weights, 
        # but technically we only want positive correlation with acceptance.
        # If a coeff is negative, we'll floor it at a small positive minimum.
        coeffs = model.coef_[0]
        
        # Softmax or Normalize to sum to 1
        pos_coeffs = np.maximum(coeffs, 0.01) # Avoid zero or negative weights
        normalized_weights = pos_coeffs / pos_coeffs.sum()
        
        return {
            'w1': float(normalized_weights[0]), # TextSim
            'w2': float(normalized_weights[1]), # MBTI
            'w3': float(normalized_weights[2])  # Location
        }

if __name__ == "__main__":
    fl = AdaptiveFeedbackLoop()
    # Test on a generic user. Since feedback is somewhat random, weights will shift.
    # U050 has some feedback.
    print(f"Optimal weights for U050: {fl.get_optimal_weights('U050')}")
