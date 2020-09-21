# developed for predicting the goals scored in football matches using machine learning

import pandas as pd
import math
import numpy as np

# reading the database
df = pd.read_csv("D1_16.csv",encoding="utf-8")

# creating the "date" column for analyzing previous matches
df["date"] = pd.to_datetime(df["Date"],format="%d/%m/%y")

# merging the databases
for i in range(17,21):

    path = 'D1_' + str(i) + '.csv'
    dfcon = pd.read_csv(path,encoding="utf-8")
    print(dfcon.head())
    print(dfcon.shape)

    if(i>18):
        dfcon["date"] = pd.to_datetime(dfcon["Date"], format="%d/%m/%Y")
    else:
        dfcon["date"] = pd.to_datetime(dfcon["Date"], format="%d/%m/%y")

    df = pd.concat([df, dfcon], ignore_index=True, sort=False)


home_goal_out = list(df["FTHG"])
away_goal_out = list(df["FTAG"])

home_team = list(df["HomeTeam"])
away_team = list(df["AwayTeam"])

date = list(df["date"])

inputs = []
outputs = []

leng = len(date)

n = 4

for i in(range(leng)):

    sub = df[(df['date'] < date[i]) & (df["HomeTeam"] == home_team[i]) ]

    sub = sub.tail(n=n)

    home_shots = sub["HS"].mean()
    home_shots_t = sub["HST"].mean()
    home_corner = sub["HC"].mean()
    home_foul = sub["HF"].mean()

    home_goal = sub["FTHG"].mean()

    sub = df[(df['date'] < date[i]) & (df["AwayTeam"] == away_team[i])]

    sub = sub.tail(n=n)

    away_shots = sub["AS"].mean()
    away_shots_t = sub["AST"].mean()
    away_corner = sub["AC"].mean()
    away_foul = sub["AF"].mean()

    away_goal = sub["FTAG"].mean()


    if((not math.isnan(home_shots)) and (not math.isnan(away_shots))):

        total = home_goal_out[i] + away_goal_out[i]

        if (total > 2):
            out = 1
        else:
            out = 0

        """
        if(total<2):
            out=0
        elif(total>1 and total<4):
            out=1
        elif(total>3 and total<6):
            out=2
        else:
            out=3
        """

        outputs.append(out)

        inputs.append([
            home_shots,home_shots_t,home_corner,home_goal,home_foul,
            away_shots,away_shots_t,away_corner,away_goal,away_foul ])

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score, train_test_split

# scaling
scaler = preprocessing.StandardScaler()

# splitting the database into the train and test
X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(np.array(inputs)), np.array(outputs), test_size=0.2)

from sklearn.linear_model import LogisticRegression

# LR model
clf = LogisticRegression()
model = clf.fit(X_train,y_train)
out = clf.score(X_test,y_test)
print(out)

cv_scores = cross_val_score(clf, inputs, outputs, cv=5)
print(cv_scores.mean())

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Neural networks model

model = Sequential()
model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=1,
          epochs=4,verbose=2)

score, acc = model.evaluate(X_test, y_test)
print(acc)

# svm model

from sklearn import svm
C = 1.0
svc = svm.SVC(kernel = "rbf" , C=C)
svc.fit(X_train,y_train)
out = svc.score(X_test,y_test)
print(out)

svc = svm.SVC(kernel = "poly" , C=C)
svc.fit(X_train,y_train)
out = svc.score(X_test,y_test)
print(out)