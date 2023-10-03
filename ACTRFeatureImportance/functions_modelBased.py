import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import functions_preprocessing

def modelBased_randomForest():
    # Data Loading and Preprocessing
    df = functions_preprocessing.load_data()
    X_train, _, y_train, _ = functions_preprocessing.split_data(df)  # Using _ for discarded return values
    X_train, _, feature_names = functions_preprocessing.one_hot_encode_and_scale(X_train, None, df)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(X_train, y_train)

    # Feature Importances Extraction
    def get_original_feature_name(encoded_name):
        if 'encoder__' in encoded_name: return encoded_name.split('__')[1].split('_')[0]
        if 'remainder__' in encoded_name: return encoded_name.replace('remainder__', '')
        return encoded_name

    aggregated_importances = {}
    for label, importance in zip(feature_names, model.feature_importances_):
        original_name = get_original_feature_name(label)
        aggregated_importances[original_name] = aggregated_importances.get(original_name, 0) + importance
    
    df_scores = pd.DataFrame(list(aggregated_importances.values()), columns=['Importance'])
    df_col = pd.DataFrame(list(aggregated_importances.keys()), columns=['Feature'])
    
    return df_scores, df_col
