import csv
import datetime
import random
from scoring_engine import ScoringEngine

def inject_test_bias():
    filename = 'e:/Profile matching/feedback.csv'
    engine = ScoringEngine()
    users_df = engine.users_df
    
    test_user = "U001"
    test_user_mbti = users_df.loc[test_user]['mbti']
    
    # We want to create 15 interactions for U001.
    # They will ACCEPT if the other person has a highly compatible MBTI.
    # They will REJECT if the MBTI match is poor, regardless of text sim.
    
    new_rows = []
    current_time = datetime.datetime.now()
    
    other_users = users_df.index.tolist()
    other_users.remove(test_user)
    
    # Randomly select 15 users to evaluate
    sample_users = random.sample(other_users, 15)
    
    for m_id in sample_users:
        score_data = engine.get_score(test_user, m_id)
        mbti_score = score_data['mbti_match']
        
        # Biased rule: Accept if MBTI Match >= 0.75, else Reject
        action = 1 if mbti_score >= 0.75 else 0
        
        current_time += datetime.timedelta(hours=1)
        timestamp_str = current_time.strftime("%d-%m-%Y %H:%M")
        
        row = {
            'user_id': test_user,
            'matched_user_id': m_id,
            'action': str(action),
            'timestamp': timestamp_str
        }
        new_rows.append(row)
        
    with open(filename, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['user_id', 'matched_user_id', 'action', 'timestamp'])
        for row in new_rows:
            writer.writerow(row)
            
    print("Injected 15 biased interactions for U001.")

if __name__ == "__main__":
    inject_test_bias()
