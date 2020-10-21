from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    goal = request.form['goal']
    blurb = request.form['blurb']
    creation = request.form['creation']
    launch = request.form['launch']
    deadline = request.form['deadline']
    category = request.form['category']
    country = request.form['country']
    launchTime = request.form['launchTime']
    deadlineTime = request.form['deadlineTime']

    arr = np.array(
        [[goal, blurb, creation, launch, deadline, category, country, launchTime, deadlineTime]])

    df = pd.DataFrame(data=arr, index=["input"], columns=[
        "goal", "blurb", "creation", "launch", "deadline", 'category', 'country', 'launchTime', 'deadlineTime'])
    df[['creation', 'deadline', 'launch']] = df[['creation', 'deadline',
                                                 'launch']].apply(pd.to_datetime)  # if conversion required
    df['launch_to_deadline'] = (df['deadline'] - df['launch']).dt.days
    df['creation_to_launch'] = (df['launch'] - df['creation']).dt.days
    df = df.drop(columns=['creation', 'launch', 'deadline'])
    df["goal"] = df['goal'].astype('int')
    df["blurb"] = df['blurb'].astype('int')

    if category == 'Art':
        cat_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Comics':
        cat_arr = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Crafts':
        cat_arr = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Dance':
        cat_arr = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Design':
        cat_arr = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Fashion':
        cat_arr = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Film':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Food':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif category == 'Games':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif category == 'Journalism':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif category == 'Music':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif category == 'Photography':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif category == 'Publishing':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif category == 'Technology':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    elif category == 'Theater':
        cat_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    df_cat = pd.DataFrame(data=cat_arr, index=["input"], columns=[
        'category_art',
        'category_comics',
        'category_crafts',
        'category_dance',
        'category_design',
        'category_fashion',
        'category_film & video',
        'category_food',
        'category_games',
        'category_journalism',
        'category_music',
        'category_photography',
        'category_publishing',
        'category_technology',
        'category_theater'])

    if country == 'Australia':
        country_arr = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Austria':
        country_arr = np.array(
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Belgium':
        country_arr = np.array(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Canada':
        country_arr = np.array(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Denmark':
        country_arr = np.array(
            [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'France':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Germany':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Hong Kong':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Ireland':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Italy':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Japan':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Luxembourg':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Mexico':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'New Zealand':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Norway':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif country == 'Singapore':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif country == 'Spain':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif country == 'Sweden':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif country == 'Switzerland':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif country == 'the Netherlands':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif country == 'the United Kingdom':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    elif country == 'the United States':
        country_arr = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    df_country = pd.DataFrame(data=country_arr, index=["input"], columns=[
        'country_displayable_name_Australia',
        'country_displayable_name_Austria',
        'country_displayable_name_Belgium',
        'country_displayable_name_Canada',
        'country_displayable_name_Denmark',
        'country_displayable_name_France',
        'country_displayable_name_Germany',
        'country_displayable_name_Hong Kong',
        'country_displayable_name_Ireland',
        'country_displayable_name_Italy',
        'country_displayable_name_Japan',
        'country_displayable_name_Luxembourg',
        'country_displayable_name_Mexico',
        'country_displayable_name_New Zealand',
        'country_displayable_name_Norway',
        'country_displayable_name_Singapore',
        'country_displayable_name_Spain',
        'country_displayable_name_Sweden',
        'country_displayable_name_Switzerland',
        'country_displayable_name_the Netherlands',
        'country_displayable_name_the United Kingdom',
        'country_displayable_name_the United States'])

# launch time
    if launchTime == '10am-12pm':
        launchT_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '10pm-12am':
        launchT_arr = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '12am-2am':
        launchT_arr = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '12pm-2pm':
        launchT_arr = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '2am-4am':
        launchT_arr = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '2pm-4pm':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif launchTime == '4am-6am':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif launchTime == '4pm-6pm':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif launchTime == '6am-8am':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif launchTime == '6pm-8pm':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif launchTime == '8am-10am':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    elif launchTime == '8pm-10pm':
        launchT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    df_launchT = pd.DataFrame(data=launchT_arr, index=["input"], columns=['launch_time_10am-12pm',
                                                                          'launch_time_10pm-12am',
                                                                          'launch_time_12am-2am',
                                                                          'launch_time_12pm-2pm',
                                                                          'launch_time_2am-4am',
                                                                          'launch_time_2pm-4pm',
                                                                          'launch_time_4am-6am',
                                                                          'launch_time_4pm-6pm',
                                                                          'launch_time_6am-8am',
                                                                          'launch_time_6pm-8pm',
                                                                          'launch_time_8am-10am',
                                                                          'launch_time_8pm-10pm'])

# deadline time
    if deadlineTime == '10am-12pm':
        deadlineT_arr = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '10pm-12am':
        deadlineT_arr = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '12am-2am':
        deadlineT_arr = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '12pm-2pm':
        deadlineT_arr = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '2am-4am':
        deadlineT_arr = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '2pm-4pm':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif deadlineTime == '4am-6am':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif deadlineTime == '4pm-6pm':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif deadlineTime == '6am-8am':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif deadlineTime == '6pm-8pm':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif deadlineTime == '8am-10am':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    elif deadlineTime == '8pm-10pm':
        deadlineT_arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    df_deadlineT = pd.DataFrame(data=deadlineT_arr, index=["input"], columns=['deadline_time_10am-12pm',
                                                                              'deadline_time_10pm-12am',
                                                                              'deadline_time_12am-2am',
                                                                              'deadline_time_12pm-2pm',
                                                                              'deadline_time_2am-4am',
                                                                              'deadline_time_2pm-4pm',
                                                                              'deadline_time_4am-6am',
                                                                              'deadline_time_4pm-6pm',
                                                                              'deadline_time_6am-8am',
                                                                              'deadline_time_6pm-8pm',
                                                                              'deadline_time_8am-10am',
                                                                              'deadline_time_8pm-10pm'])

    df = df.merge(df_cat, left_index=True, right_index=True)
    df = df.drop(columns=['category'])
    df = df.merge(df_country, left_index=True, right_index=True)
    df = df.drop(columns=['country'])
    df = df.merge(df_launchT, left_index=True, right_index=True)
    df = df.drop(columns=['launchTime'])
    df = df.merge(df_deadlineT, left_index=True, right_index=True)
    df = df.drop(columns=['deadlineTime'])

    X_train_scaled = scaler.transform(df)

    pred = model.predict(X_train_scaled)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
