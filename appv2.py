import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df.head()

df["date"] = pd.to_datetime(df["date"])
df_train = df[df["date"] < "2015-06-23"]
df_train.head()

politics_articles = df_train[df_train["category"] == "POLITICS"]
poli_count = politics_articles.groupby("date").size().reset_index(name="Number of Articles")
poli_count

X = pd.to_numeric(poli_count["date"]).values.reshape(-1,1)
Y = poli_count["Number of Articles"].values

model1 = LinearRegression()
model1.fit(X,Y)
y_pred = model1.predict(X)

plt.figure()
plt.scatter(poli_count["date"], poli_count["Number of Articles"], label="Training Data, 2014-2015")
plt.plot(poli_count["date"], y_pred, color="red", label="Linear Regression")
plt.title("Politics articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()

df_test = df[df["date"] > "2015-06-23"]
df_test.head()

politics_articles_test = df_test[df_test["category"] == "POLITICS"]
poli_count_test = politics_articles_test.groupby("date").size().reset_index(name="Number of Articles")
poli_count_test

X2 = pd.to_numeric(poli_count_test["date"]).values.reshape(-1,1)
Y2 = poli_count_test["Number of Articles"].values

model2 = LinearRegression()
model2.fit(X2,Y2)
y2_pred = model2.predict(X2)

plt.figure()
plt.scatter(poli_count_test["date"], poli_count_test["Number of Articles"], label="Training Data, 2014-2015")
plt.plot(poli_count["date"], y_pred, color="red", label="Linear Regression")
plt.plot(poli_count_test["date"], y2_pred, color="green", label="Linear Regression 2")
plt.title("Politics articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()

accuracy = r2_score(Y2,y2_pred)
accuracy

sports_articles = df_train[df_train["category"] == "SPORTS"]
sport_count = sports_articles.groupby("date").size().reset_index(name="Number of Articles")
sport_count

X3 = pd.to_numeric(poli_count["date"]).values.reshape(-1,1)
Y3 = poli_count["Number of Articles"].values

model3 = LinearRegression()
model3.fit(X3,Y3)
y3_pred = model3.predict(X3)

plt.figure()
plt.scatter(sport_count["date"], sport_count["Number of Articles"], label="Training Data, 2014-2015")
plt.plot(sport_count["date"], y3_pred, color="red", label="Linear Regression")
plt.title("Sports articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()

sports_articles_test = df_test[df_test["category"] == "SPORTS"]
sports_count_test = sports_articles_test.groupby("date").size().reset_index(name="Number of Articles")
sports_count_test

X4 = pd.to_numeric(sports_count_test["date"]).values.reshape(-1,1)
Y4 = sports_count_test["Number of Articles"].values

model4 = LinearRegression()
model4.fit(X4,Y4)
y4_pred = model4.predict(X4)

plt.figure()
plt.scatter(poli_count_test["date"], poli_count_test["Number of Articles"], label="Training Data, Before Verizon")
plt.plot(sport_count["date"], y3_pred, color="red", label="Linear Regression")
plt.plot(sports_count_test["date"], y4_pred, color="green", label="Linear Regression 2")
plt.title("Sports articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()

enter_articles = df_train[df_train["category"] == "ENTERTAINMENT"]
enter_count = enter_articles.groupby("date").size().reset_index(name="Number of Articles")
enter_count

X5 = pd.to_numeric(enter_count["date"]).values.reshape(-1,1)
Y5 = enter_count["Number of Articles"].values

model5 = LinearRegression()
model5.fit(X5,Y5)
y5_pred = model5.predict(X5)

plt.figure()
plt.scatter(enter_count["date"], enter_count["Number of Articles"], label="Training Data, 2014-2015")
plt.plot(enter_count["date"], y5_pred, color="red", label="Linear Regression")
plt.title("Entertainment articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()

enter_articles_test = df_test[df_test["category"] == "ENTERTAINMENT"]
enter_count_test = enter_articles_test.groupby("date").size().reset_index(name="Number of Articles")
enter_count_test

X6 = pd.to_numeric(enter_count_test["date"]).values.reshape(-1,1)
Y6 = enter_count_test["Number of Articles"].values

model6 = LinearRegression()
model6.fit(X6,Y6)
y6_pred = model6.predict(X6)

plt.figure()
plt.scatter(enter_count_test["date"], enter_count_test["Number of Articles"], label="Training Data, Before Verizon")
plt.plot(enter_count["date"], y5_pred, color="red", label="Linear Regression")
plt.plot(enter_count_test["date"], y6_pred, color="green", label="Linear Regression 2")
plt.title("Entertainment articles per day")
plt.xlabel("Date")
plt.ylabel("# of articles")
plt.show()