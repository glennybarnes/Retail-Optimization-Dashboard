
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("../data/SampleSuperstore.csv")

# Preprocessing
df['Is_Loss'] = (df['Profit'] < 0).astype(int)

# Sidebar filters
st.sidebar.header("Filters")
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
category = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
discount_range = st.sidebar.slider("Discount Range", float(df['Discount'].min()), float(df['Discount'].max()), (0.0, 0.8))

# Filtered data
filtered_df = df[(df['Region'].isin(region)) & 
                 (df['Category'].isin(category)) & 
                 (df['Discount'] >= discount_range[0]) & 
                 (df['Discount'] <= discount_range[1])]

st.title("📊 Superstore Real-Time Dashboard")

# Metrics
st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
st.metric("Average Discount", f"{filtered_df['Discount'].mean():.2%}")

# Profit by Sub-Category
st.subheader("Profit by Sub-Category")
fig1, ax1 = plt.subplots()
subcat_profit = filtered_df.groupby('Sub-Category')['Profit'].sum().sort_values()
sns.barplot(x=subcat_profit.values, y=subcat_profit.index, ax=ax1)
st.pyplot(fig1)

# Discount vs Profit Scatter
st.subheader("Discount vs Profit")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_df, x='Discount', y='Profit', hue='Category', ax=ax2)
st.pyplot(fig2)

# Predictive Loss Alerts
st.subheader("Loss Risk Prediction Alerts")
features = ['Sales', 'Discount', 'Quantity']
X = df[features]
y = df['Is_Loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
filtered_df['Predicted_Loss'] = model.predict(filtered_df[features])
alerts = filtered_df[filtered_df['Predicted_Loss'] == 1][['Sub-Category', 'Sales', 'Discount', 'Profit', 'Region', 'Segment']]
st.dataframe(alerts)
