# Importing libraries
from flask import Flask, request, render_template
from model import *

# Creating a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():

    user = request.form['username'].lower()
    recommendations = GetSentimentRecommendations(user)

    if(not(recommendations is None)):
        print(f"Retrieving top {len(recommendations)} recommendations")
        print(recommendations)

        return render_template("index.html", user_name=user, column_names=recommendations.columns.values, row_data=list(recommendations.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message=f"Invalid username: \"{user}\"")

if __name__ == '__main__':
    app.run()