import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Fuel Station AI Dashboard",
    layout="wide"
)
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
</style>
""", unsafe_allow_html=True)
st.title(" Fuel Station AI Dashboard")
with st.spinner("Loading AI Dashboard..."):
    st.success("Dashboard Ready")
st.success("AI Powered Fuel Station Analytics System")

df = pd.read_csv("Fuel_Station_Dataset.csv")

st.sidebar.title("Dashboard Filter")

city = st.sidebar.selectbox(
    "Choose City",
    df['City'].unique()
)

filtered_df = df[df['City'] == city]


st.subheader("Dataset Preview")

st.dataframe(filtered_df.head())

total_revenue = filtered_df['Revenue'].sum()
total_profit = filtered_df['Profit'].sum()
total_customers = filtered_df['Customer_Count'].sum()
avg_rating = round(filtered_df['Customer_Rating'].mean(), 2)
st.download_button(
    " Download Full Dataset",
    df.to_csv(index=False),
    file_name="fuel_station_full_data.csv",
    mime="text/csv"
)
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Today's Revenue",
    f"₹{total_revenue:,.0f}",
    "+12%"
)
col2.metric(
    "Total Profit",
    f"₹{total_profit:,.0f}"
)

col3.metric(
    "Customers",
    f"{total_customers:,}"
)

col4.metric(
    "Average Rating",
    avg_rating
)
from fpdf import FPDF

def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=" Fuel Station Analytics Report", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Total Sales: {data['Sales'].sum()}", ln=True)
    pdf.cell(200, 10, txt=f"Total Litres: {data['Litres'].sum()}", ln=True)
    pdf.cell(200, 10, txt=f"Average Price: {data['Price'].mean():.2f}", ln=True)

    file_path = "fuel_report.pdf"
    pdf.output(file_path)
    return file_path

best_city = df.groupby('City')['Profit'].sum().idxmax()

st.info(f" Best Performing City: {best_city}")
st.subheader(" AI Profit Prediction")

X = filtered_df[['Revenue']]
y = filtered_df['Profit']

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    filtered_df['Revenue'],
    filtered_df['Profit'],
    color='gold',
    label='Actual Profit'
)

ax.plot(
    filtered_df['Revenue'],
    predictions,
    color='black',
    linewidth=3,
    label='Predicted Profit'
)

ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")
ax.set_title("AI Profit Prediction")
ax.legend()

st.pyplot(fig)


st.subheader("Predict Profit")

user_input = st.number_input(
    "Enter Revenue Amount",
    min_value=0
)

prediction = model.predict([[user_input]])

st.success(
    f"Predicted Profit: ₹{prediction[0]:,.2f}"
)

st.subheader("Revenue Distribution")

fig, ax = plt.subplots(figsize=(8,5))

ax.hist(
    filtered_df['Revenue'],
    bins=20,
    color='magenta'
)

ax.set_title("Revenue Distribution")

st.pyplot(fig)

st.subheader("Fuel Sold by Fuel Type")

fig, ax = plt.subplots(figsize=(8,5))

colors = ['yellow','blue','cyan']

filtered_df.groupby('Fuel_Type')['Fuel_Sold_Liters'].sum().plot(
    kind='bar',
    color=colors,
    ax=ax
)

ax.set_title("Fuel Sold by Fuel Type")
ax.set_xlabel("Fuel Type")
ax.set_ylabel("Liters")

st.pyplot(fig)


st.subheader("Payment Mode Distribution")

fig, ax = plt.subplots(figsize=(7,7))

filtered_df['Payment_Mode'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)


st.subheader("Revenue Trend")

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    filtered_df['Revenue'].head(50),
    color='brown'
)

ax.set_title("Revenue Trend")

st.pyplot(fig)

st.subheader("Revenue vs Profit")

fig, ax = plt.subplots(figsize=(8,5))

colors = ['gold', 'silver']

for i in range(len(filtered_df)):
    ax.scatter(
        filtered_df['Revenue'].iloc[i],
        filtered_df['Profit'].iloc[i],
        color=colors[i % 2]
    )

ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")
ax.set_title("Revenue vs Profit")

st.pyplot(fig)
st.subheader(" Fuel Distribution")

fig2, ax2 = plt.subplots()
ax2.bar(filtered_df["Fuel_Type"], filtered_df["Profit"])
ax2.set_title("Fuel Type vs Profit")
st.pyplot(fig2)


st.subheader("Revenue Box Plot")

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    y=filtered_df['Revenue'],
    color='red',
    ax=ax
)

ax.set_title("Revenue Box Plot")

st.pyplot(fig)

st.subheader("City Count")

fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(
    x=filtered_df['City'],
    color='violet',
    ax=ax
)

ax.set_title("City Count")

st.pyplot(fig)


st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    filtered_df.corr(numeric_only=True),
    annot=True,
    ax=ax
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)

st.subheader("Radar Chart")

categories = [
    'Revenue',
    'Profit',
    'Customer_Count',
    'Employee_Count',
    'Customer_Rating'
]

values = [
    filtered_df['Revenue'].mean(),
    filtered_df['Profit'].mean(),
    filtered_df['Customer_Count'].mean(),
    filtered_df['Employee_Count'].mean(),
    filtered_df['Customer_Rating'].mean() * 1000
]

N = len(categories)

angles = np.linspace(
    0,
    2 * np.pi,
    N,
    endpoint=False
).tolist()

values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(8,8),
    subplot_kw=dict(polar=True)
)

ax.plot(
    angles,
    values,
    linewidth=2,
    color='purple'
)

ax.fill(
    angles,
    values,
    color='violet',
    alpha=0.4
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.title("Fuel Station Radar Chart")

st.pyplot(fig)


st.subheader("AI Customer Clustering")

X_cluster = filtered_df[['Revenue', 'Profit']]

kmeans = KMeans(
    n_clusters=3,
    random_state=0
)

filtered_df['Cluster'] = kmeans.fit_predict(X_cluster)

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    filtered_df['Revenue'],
    filtered_df['Profit'],
    c=filtered_df['Cluster'],
    cmap='viridis',
    s=80
)

ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")
ax.set_title("AI Customer Clustering")

st.pyplot(fig)

filtered_df['Health_Score'] = (
    filtered_df['Customer_Rating'] * 20
    + filtered_df['Profit'] / 1000
)

health = int(filtered_df['Health_Score'].mean())

st.subheader("Fuel Station Health Meter")

st.progress(health // 10)

st.write(f"Overall Health Score: {health}")
if filtered_df['Profit'].mean() > 50000:
    st.info(" AI Insight: Business performance is excellent.")
else:
    st.warning(" AI Insight: Profit growth needed.")


st.subheader("Business Performance Analysis")

avg_profit = filtered_df['Profit'].mean()

fig, ax = plt.subplots(figsize=(6,4))

if avg_profit > 50000:

    ax.bar(
        ['Business Status'],
        [avg_profit],
        color='gold'
    )

    ax.text(
        0,
        avg_profit + 2000,
        'Excellent Performance',
        ha='center',
        fontsize=12
    )

else:

    ax.bar(
        ['Business Status'],
        [avg_profit],
        color='red'
    )

    ax.text(
        0,
        avg_profit + 2000,
        'Needs Improvement',
        ha='center',
        fontsize=12
    )

ax.set_title("Business Performance")
ax.set_ylabel("Average Profit")

st.pyplot(fig)


csv = filtered_df.to_csv(index=False)

st.subheader("Profit Violin Plot")

fig, ax = plt.subplots(figsize=(8,5))

sns.violinplot(
    y=filtered_df['Profit'],
    color='violet',
    ax=ax
)

st.pyplot(fig)

st.subheader("Customer Rating Swarm Plot")

fig, ax = plt.subplots(figsize=(8,5))

sns.swarmplot(
    x=filtered_df['Fuel_Type'],
    y=filtered_df['Customer_Rating'],
    ax=ax
)

st.pyplot(fig)


st.subheader("Hexbin Plot")

fig, ax = plt.subplots(figsize=(8,5))

ax.hexbin(
    filtered_df['Revenue'],
    filtered_df['Profit'],
    gridsize=20,
    cmap='viridis'
)

ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")

st.pyplot(fig)


st.subheader("Fuel Type Donut Chart")

fig, ax = plt.subplots(figsize=(7,7))

ax.pie(
    filtered_df['Fuel_Type'].value_counts(),
    labels=filtered_df['Fuel_Type'].value_counts().index,
    autopct='%1.1f%%'
)

circle = plt.Circle((0,0),0.70,color='white')

fig.gca().add_artist(circle)

st.pyplot(fig)

st.subheader("Bubble Chart")

fig, ax = plt.subplots(figsize=(8,5))

sizes = filtered_df['Customer_Count'] * 5

scatter = ax.scatter(
    filtered_df['Revenue'],
    filtered_df['Profit'],
    s=sizes,
    alpha=0.5,
    color='purple'
)

ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")

st.pyplot(fig)


st.subheader("Spiral Visualization")

theta = np.linspace(0, 8*np.pi, 500)

r = np.linspace(0, 10, 500)

x = r * np.cos(theta)

y = r * np.sin(theta)

fig, ax = plt.subplots(figsize=(8,8))

ax.plot(
    x,
    y,
    color='darkblue',
    linewidth=2
)

ax.set_aspect('equal')

st.pyplot(fig)


st.subheader("Triangle Relationship Chart")

fig, ax = plt.subplots(figsize=(8,7))

x = [0, 1, 2, 0]

y = [0, 2, 0, 0]

ax.plot(
    x,
    y,
    color='red',
    linewidth=3
)

ax.fill(
    x,
    y,
    color='orange',
    alpha=0.5
)

ax.text(0, 0, "Revenue")

ax.text(1, 2, "Profit")

ax.text(2, 0, "Customers")

ax.axis('off')

st.pyplot(fig)

st.subheader("Polar Profit Chart")

theta = np.linspace(
    0,
    2*np.pi,
    len(filtered_df.head(20))
)

r = filtered_df['Profit'].head(20)

fig = plt.figure(figsize=(8,8))

ax = plt.subplot(111, projection='polar')

ax.plot(
    theta,
    r,
    color='gold',
    linewidth=2
)

ax.fill(
    theta,
    r,
    alpha=0.3,
    color='orange'
)

st.pyplot(fig)


st.subheader("Circular Bar Plot")

fuel = filtered_df['Fuel_Type'].value_counts()

angles = np.linspace(
    0,
    2*np.pi,
    len(fuel),
    endpoint=False
)

fig = plt.figure(figsize=(8,8))

ax = plt.subplot(111, polar=True)

bars = ax.bar(
    angles,
    fuel.values,
    width=0.5,
    color=['gold','silver','brown']
)

ax.set_xticks(angles)

ax.set_xticklabels(fuel.index)

st.pyplot(fig)


st.subheader("Spider Web Chart")

labels = [
    'Revenue',
    'Profit',
    'Expenses',
    'Customers',
    'Rating'
]

stats = [
    filtered_df['Revenue'].mean()/100,
    filtered_df['Profit'].mean()/100,
    filtered_df['Daily_Expense'].mean()/100,
    filtered_df['Customer_Count'].mean(),
    filtered_df['Customer_Rating'].mean()*20
]

angles = np.linspace(
    0,
    2*np.pi,
    len(labels),
    endpoint=False
).tolist()

stats += stats[:1]

angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(8,8),
    subplot_kw=dict(polar=True)
)

ax.plot(
    angles,
    stats,
    color='blue',
    linewidth=2
)

ax.fill(
    angles,
    stats,
    color='lightblue',
    alpha=0.4
)

ax.set_xticks(angles[:-1])

ax.set_xticklabels(labels)

st.pyplot(fig)

st.subheader("Wave Plot")

x = np.linspace(0, 20, 500)

y = np.sin(x)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    x,
    y,
    linewidth=3,
    color='blue'
)

st.pyplot(fig)

st.subheader("Customer Funnel")

stages = ['Visitors','Customers','Regulars']

values = [1000, 700, 400]

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    stages,
    values,
    color=['gold','silver','brown']
)

st.pyplot(fig)

st.success(" Fuel Station AI Dashboard Loaded Successfully")
def create_charts(data):

    os.makedirs("charts", exist_ok=True)

    # 1. Revenue vs Profit
    plt.figure()
    plt.scatter(data["Revenue"], data["Profit"])
    plt.title("Revenue vs Profit")
    plt.savefig("charts/chart1.png")
    plt.close()

    # 2. Fuel Type Sales
    if "Fuel_Type" in data.columns:
        plt.figure()
        data.groupby("Fuel_Type")["Revenue"].sum().plot(kind="bar")
        plt.title("Fuel Type Revenue")
        plt.savefig("charts/chart2.png")
        plt.close()

    # 3. Customer Count
    if "Customer_Count" in data.columns:
        plt.figure()
        data["Customer_Count"].plot(kind="hist")
        plt.title("Customer Distribution")
        plt.savefig("charts/chart3.png")
        plt.close()

    # 4. Profit Trend
    plt.figure()
    data["Profit"].plot()
    plt.title("Profit Trend")
    plt.savefig("charts/chart4.png")
    plt.close()

    # 5. Payment Mode
    if "Payment_Mode" in data.columns:
        plt.figure()
        data["Payment_Mode"].value_counts().plot(kind="pie")
        plt.title("Payment Mode Distribution")
        plt.savefig("charts/chart5.png")
        plt.close()
def generate_pdf(data):

    pdf = FPDF()

    # ================= PAGE 1 - TITLE =================
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "FUEL STATION ANALYTICS REPORT", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Total Records: {len(data)}", ln=True)

    pdf.ln(5)

    # ================= PAGE 2 =================
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Revenue vs Profit", ln=True)

    pdf.image("charts/chart1.png", w=180)

    # ================= PAGE 3 =================
    pdf.add_page()
    pdf.cell(200, 10, "Fuel Type Revenue", ln=True)
    pdf.image("charts/chart2.png", w=180)

    # ================= PAGE 4 =================
    pdf.add_page()
    pdf.cell(200, 10, "Customer Distribution", ln=True)
    pdf.image("charts/chart3.png", w=180)

    # ================= PAGE 5 =================
    pdf.add_page()
    pdf.cell(200, 10, "Profit Trend", ln=True)
    pdf.image("charts/chart4.png", w=180)

    # ================= PAGE 6 =================
    pdf.add_page()
    pdf.cell(200, 10, "Payment Mode Analysis", ln=True)
    pdf.image("charts/chart5.png", w=180)

    # SAVE PDF
    file_path = "fuel_dashboard_report.pdf"
    pdf.output(file_path)

    return file_path
if st.button(" Generate Full Dashboard PDF (5 Charts)"):

    create_charts(filtered_df)

    file_path = generate_pdf(filtered_df)

    with open(file_path, "rb") as f:
        st.download_button(
            " Download Full Report",
            data=f,
            file_name="fuel_dashboard_report.pdf",
            mime="application/pdf"
        )

    st.success("Multi-page PDF Ready!")
st.subheader(" Key Insights")

st.info("• Fuel sales are visualized based on selected type")
st.info("• Dashboard helps track revenue performance")
st.info("• Use filters to analyze specific fuel category")