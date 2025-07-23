
# 🛍️ Retail Optimization Dashboard with Predictive Insights

This project presents an end-to-end data analytics solution for a retail business using the Superstore dataset. The goal is to optimize discount strategies by analyzing their impact on profitability and to predict loss-making transactions in real-time.

## 📌 Objectives
- Analyze how discounts affect profits across categories and sub-categories.
- Identify high-risk transactions that lead to losses.
- Build a Streamlit dashboard to visualize trends and provide actionable business insights.

## 🧰 Tools Used
- **Python**: Data analysis and machine learning
- **Pandas, Matplotlib, Seaborn**: Data wrangling and visualization
- **Scikit-learn**: Predictive modeling
- **Streamlit**: Real-time interactive dashboard

## 📊 Features of the Dashboard
- Filter data by region, category, and discount range
- View key metrics: Total Sales, Total Profit, Average Discount
- Analyze profit distribution across sub-categories
- Visualize Discount vs Profit with interactive scatter plots
- Flag transactions with high loss risk using predictive modeling

## 📁 Project Structure
```
Retail-Optimization-Dashboard/
│
├── data/
│   └── SampleSuperstore.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── app/
│   └── retail_dashboard.py
│
├── visuals/
│   └── charts and screenshots
│
├── README.md
└── requirements.txt
```

## 🚀 How to Run
1. Clone this repo
2. Ensure your environment includes required packages
3. Launch the dashboard:
```bash
streamlit run app/retail_dashboard.py
```

## 📸 Screenshots
*Add visuals from your dashboard here for visual impact.*

## 📈 Business Impact
This tool empowers decision-makers with real-time insights into the profitability of discounting strategies. It flags potential loss scenarios before they happen, improving financial health and promotional targeting.
