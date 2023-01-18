# Importing libraries
import pandas as pd
import numpy as np
import pickle

# Constants
root_path = "pickle//"
XGBoost_model = "SentimentClassificationXGBoostModel.pkl"
tfidfVectorizer = "tfidfVectorizer.pkl"
recommender = "UserBasedRecommendationModel.pkl"
cleanData = "CleanedData.pkl"

# Sentiment based recommendations
def GetSentimentRecommendations(user):

    XGBoostModel = pickle.load(open(root_path + XGBoost_model, 'rb'))
    vectorizer = pd.read_pickle(root_path + tfidfVectorizer)
    UserBasedRecommendationModel = pickle.load(open(root_path + recommender, 'rb'))
    data = pd.read_csv("sample30.csv")
    cleanedData = pickle.load(open(root_path + cleanData, 'rb'))

    if (user not in UserBasedRecommendationModel.index):
        print(f"Username {user} does not exist.")
        return None
    
    else:
        # Get the product recommendations using the trained ML model 'UserBasedRecommendationModel'
        recommendations = list(UserBasedRecommendationModel.loc[user].sort_values(ascending=False)[0:20].index)
        filteredData = cleanedData[cleanedData.id.isin(recommendations)]

        # transform the input data using saved tf-idf vectorizer
        X = vectorizer.transform(filteredData["reviews_Lemmatext"].values.astype(str))
        
        # Predict using the saved XGBoost model
        filteredData["predicted_sentiment"] = XGBoostModel.predict(X)
        temp = filteredData[['id', 'predicted_sentiment']]
        temp_grouped = temp.groupby('id', as_index=False).count()
        temp_grouped["positive_review_count"] = temp_grouped.id.apply(lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['positive_sentiment_percent'] = np.round(temp_grouped["positive_review_count"] / temp_grouped["total_review_count"] * 100, 2)

        sorted_products = temp_grouped.sort_values('positive_sentiment_percent', ascending=False)[0:5]
        
        return pd.merge(data, sorted_products, on="id")[
            ["name", "brand", "positive_sentiment_percent"]].drop_duplicates().sort_values(
            ['positive_sentiment_percent', 'name'], ascending=[False, True])
        