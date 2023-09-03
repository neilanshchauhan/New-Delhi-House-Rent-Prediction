from flask import Flask, render_template,request
import pandas as pd
import pickle
import sklearn 

app = Flask(__name__, template_folder='templates')
df = pd.read_csv("Cleaned_Data.csv")
pipe = pickle.load(open('GBoostModel.pkl','rb'))

@app.route("/")

def index():
    locations = sorted(df['localityName'].unique())
    return render_template('index.html',locations=locations)

@app.route("/predict",methods = ['POST'])




def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    sq_ft = request.form.get('sq_ft')
    

    df_loc = df[df['localityName']==location]
    price_per_sqft = df_loc['price'].mean() / df_loc['size_sq_ft'].mean()


    # create input DataFrame
    input_df = pd.DataFrame([[location, sq_ft, bhk, price_per_sqft]],
                            columns=['localityName', 'size_sq_ft', 'bedrooms', 'price_per_sqft'])

    # make prediction
    prediction = int(pipe.predict(input_df)[0])

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)