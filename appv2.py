import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)


# Load data
@st.cache_data
def load_data():
    return pd.read_json("News_Category_Dataset_v3.json", lines=True)

df = load_data()

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Split data into training and test sets based on date
df_train = df[df["date"] < "2015-06-23"]
df_test = df[df["date"] > "2015-06-23"]

# Define function to plot data
def plot_data(df, category, title):
    articles = df[df["category"] == category]
    count = articles.groupby("date").size().reset_index(name="Number of Articles")

    X = pd.to_numeric(count["date"]).values.reshape(-1,1)
    Y = count["Number of Articles"].values

    model = LinearRegression()
    model.fit(X,Y)
    y_pred = model.predict(X)

    plt.figure()
    plt.scatter(count["date"], count["Number of Articles"], label="Training Data, 2014-2015")
    plt.plot(count["date"], y_pred, color="red", label="Linear Regression")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("# of articles")
    plt.legend()
    st.pyplot()

# Plot Politics articles
st.header("Politics Articles per Day")
plot_data(df_train, "POLITICS", "Politics Articles per Day")
plot_data(df_test, "POLITICS", "Politics Articles per Day")

# Plot Sports articles
st.header("Sports Articles per Day")
plot_data(df_train, "SPORTS", "Sports Articles per Day")
plot_data(df_test, "SPORTS", "Sports Articles per Day")

# Plot Entertainment articles
st.header("Entertainment Articles per Day")
plot_data(df_train, "ENTERTAINMENT", "Entertainment Articles per Day")
plot_data(df_test, "ENTERTAINMENT", "Entertainment Articles per Day")
