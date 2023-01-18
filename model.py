import pandas as pd
import numpy as np
import pickle

ROOT_PATH = "pickle//"
MODEL_NAME = "SentimentClassificationXGBoostModel.pkl"
VECTORIZER = "tfidfVectorizer.pkl"
RECOMMENDER = "UserBasedRecommendationModel.pkl"
CLEANED_DATA = "CleanedData.pkl"

def GetSentimentRecommendations(user):

    XGBoostModel = pickle.load(open(ROOT_PATH + MODEL_NAME, 'rb'))
    vectorizer = pd.read_pickle(ROOT_PATH + VECTORIZER)
    UserBasedRecommendationModel = pickle.load(open(ROOT_PATH + RECOMMENDER, 'rb'))
    data = pd.read_csv("sample30.csv")
    cleanedData = pickle.load(open(ROOT_PATH + CLEANED_DATA, 'rb'))

    if (user in UserBasedRecommendationModel.index):
        # Get the product recommendations using the trained ML model 'UserBasedRecommendationModel'
        recommendations = list(UserBasedRecommendationModel.loc[user].sort_values(ascending=False)[0:20].index)
        filteredData = cleanedData[cleanedData.id.isin(recommendations)]

        # transform the input data using saved tf-idf vectorizer
        X = vectorizer.transform(filteredData["reviews_text_cleaned"].values.astype(str))
        
        # Predict using the saved XGBoost model
        filteredData["predicted_sentiment"] = XGBoostModel.predict(X)
        temp = filteredData[['id', 'predicted_sentiment']]
        temp_grouped = temp.groupby('id', as_index=False).count()
        temp_grouped["positive_review_count"] = temp_grouped.id.apply(lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['positive_sentiment_percent'] = np.round(temp_grouped["positive_review_count"] / temp_grouped["total_review_count"] * 100, 2)

        sorted_products = temp_grouped.sort_values('positive_sentiment_percent', ascending=False)[0:5]
        
        return pd.merge(data, sorted_products, on="id")[
            ["name", "brand", "manufacturer", "positive_sentiment_percent"]].drop_duplicates().sort_values(
            ['positive_sentiment_percent', 'name'], ascending=[False, True])

    else:
        print(f"User name {user} doesn't exist")
        return None