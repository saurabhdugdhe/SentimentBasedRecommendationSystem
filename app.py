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
    items = GetSentimentRecommendations(user)

    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)

        return render_template("index.html", user_name=user, column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message=f"Invalid username: \"{user}\"")

if __name__ == '__main__':
    app.run()