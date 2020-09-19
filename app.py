import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.
with open(f'model/personality_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        gender = flask.request.form['gender']
        age = flask.request.form['age']
        openness = flask.request.form['openness']
        neuroticism = flask.request.form['neuroticism']
        conscientiousness = flask.request.form['conscientiousness']
        agreeableness = flask.request.form['agreeableness']
        extraversion = flask.request.form['extraversion']
        gender_is_male = 0
        gender_is_female = 0
        if gender == 'Male':
            gender_is_male = 1
        elif gender == 'Female':
            gender_is_female = 1

        input_variables = pd.DataFrame([[age, openness, neuroticism, conscientiousness,
                                         agreeableness, extraversion, gender_is_male, gender_is_female]],
                                       columns=['Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness',
                                                'extraversion', 'Gender_is_Female', 'Gender_is_Male'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        if prediction == 0:
            prediction = 'Dependable'
        elif prediction == 1:
            prediction = 'Extraverted'
        elif prediction == 2:
            prediction = 'Lively'
        elif prediction == 3:
            prediction = 'Responsible'
        elif prediction == 4:
            prediction = 'Serious'

        return flask.render_template('main.html',
                                     original_input={'age': age, 'openness': openness, 'neuroticism': neuroticism,
                                                     'conscientiousness': conscientiousness,
                                                     'agreeableness': agreeableness,
                                                     'extraversion': extraversion,
                                                     'gender': gender},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.run()
