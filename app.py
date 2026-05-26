import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Fuel_Station_Dataset.csv")
print(df.head())
sns.set_style("whitegrid")
df.head()
print(df.info())
df.describe()
df.tail()
print(df.isnull().sum())
df.shape
df.mean(numeric_only=True)
df.median(numeric_only=True)
df.mode().iloc[0]
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = df[['Revenue']]
y = df['Profit']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
plt.figure(figsize=(8,5))
plt.scatter(df['Revenue'], df['Profit'],
            color='gold', label='Actual Profit')
plt.plot(df['Revenue'], predictions,
         color='black', linewidth=3,
         label='Predicted Profit')
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.title("AI Profit Prediction")
plt.legend()
plt.show()
plt.figure(figsize=(8,5))
plt.hist(df['Revenue'], bins=20,color = 'magenta')
plt.title("Revenue Distribution")
plt.show()
plt.figure(figsize=(8,5))
colors = ['yellow','blue','cyan']
df.groupby('Fuel_Type')['Fuel_Sold_Liters'].sum().plot(
    kind='bar',
    color=colors
)
plt.title("Fuel Sold by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Liters")
plt.show()
plt.figure(figsize=(7,7))
df['Payment_Mode'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Payment Mode Distribution")
plt.ylabel("")
plt.show()
plt.figure(figsize=(8,5))
plt.plot(df['Revenue'].head(50),color = 'brown')
plt.title("Revenue Trend")
plt.show()
colors = ['gold', 'silver']
plt.figure(figsize=(8,5))
for i in range(len(df)):
    plt.scatter(
        df['Revenue'][i],
        df['Profit'][i],
        color=colors[i % 2]
    )
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.title("Revenue vs Profit")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(y=df['Revenue'] ,color = 'red')
plt.title("Revenue Box Plot")
plt.show()
plt.figure(figsize=(8,5))
sns.violinplot(y=df['Profit'])
plt.title("Profit Violin Plot")
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(x=df['City'],color = 'violet')
plt.title("City Count")
plt.show()
plt.figure(figsize=(8,5))
sns.kdeplot(df['Customer_Rating'], fill=True)
plt.title("Customer Rating KDE")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(8,5))
plt.fill_between(range(len(df.head(50))), df['Revenue'].head(50),color = 'teal')
plt.title("Revenue Area Chart")
plt.show()
plt.figure(figsize=(8,5))
sns.stripplot(x=df['Fuel_Type'], y=df['Revenue'],color ='purple')
plt.title("Strip Plot")
plt.show()
plt.figure(figsize=(8,5))
sns.swarmplot(x=df['Fuel_Type'], y=df['Customer_Rating'])
plt.title("Swarm Plot")
plt.show()
grouped = df.groupby('City')[['Revenue','Profit']].mean()
grouped.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title("Stacked Bar Chart")
plt.show()
plt.figure(figsize=(8,5))
plt.step(range(len(df.head(50))), df['Profit'].head(50),color = 'red')
plt.title("Step Plot")
plt.show()
plt.figure(figsize=(8,5))
plt.stem(df['Customer_Count'].head(30))
plt.title("Stem Plot")
plt.show()
plt.figure(figsize=(7,7))
plt.pie(df['Fuel_Type'].value_counts(),
labels=df['Fuel_Type'].value_counts().index,
autopct='%1.1f%%')
circle = plt.Circle((0,0),0.70,color='white')
plt.gca().add_artist(circle)
plt.title("Fuel Type Donut Chart")
plt.show()
sns.pairplot(df[['Revenue','Profit','Customer_Count','Customer_Rating']])
plt.show()
plt.figure(figsize=(8,5))
plt.hexbin(df['Revenue'], df['Profit'], gridsize=20)
plt.title("Hexbin Plot")
plt.show()
plt.figure(figsize=(8,5))
sns.ecdfplot(df['Revenue'])
plt.title("ECDF Plot")
plt.show()
plt.figure(figsize=(8,5))
sns.rugplot(df['Profit'],color = 'red')
plt.title("Rug Plot")
plt.show()
sns.jointplot(x='Revenue', y='Profit', data=df)
plt.show()
sns.clustermap(df.corr(numeric_only=True), annot=True)
plt.show()
df.groupby(['City','Fuel_Type'])['Revenue'].mean().unstack().plot(kind='bar', figsize=(10,6))
plt.title("Multi Bar Chart")
plt.show()
plt.figure(figsize=(8,5))
df.groupby('City')['Profit'].mean().plot(kind='barh')
plt.title("Horizontal Bar Chart")
plt.show()
plt.figure(figsize=(8,5))
df['Revenue'].plot(kind='density')
plt.title("Density Plot")
plt.show()
plt.figure(figsize=(8,5))
df['Revenue'].cumsum().plot()
plt.title("Cumulative Revenue")
plt.show()
df[['Revenue','Profit','Customer_Count']].head(50).plot(subplots=True, figsize=(10,8))
plt.show()
categories = ['Revenue', 'Profit', 'Customer_Count',
              'Employee_Count', 'Customer_Rating']
values = [
    df['Revenue'].mean(),
    df['Profit'].mean(),
    df['Customer_Count'].mean(),
    df['Employee_Count'].mean(),
    df['Customer_Rating'].mean()*1000
]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values += values[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, color='purple')
ax.fill(angles, values, color='violet', alpha=0.4)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title("Fuel Station Radar Chart", size=15)
plt.show()
theta = np.linspace(0, 8*np.pi, 500)
r = np.linspace(0, 10, 500)
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.figure(figsize=(8,8))
plt.plot(x, y, color='darkblue', linewidth=2)
plt.title("Spiral Visualization")
plt.axis('equal')
plt.show()
plt.figure(figsize=(8,7))
x = [0, 1, 2, 0]
y = [0, 2, 0, 0]
plt.plot(x, y, color='red', linewidth=3)
plt.fill(x, y, color='orange', alpha=0.5)
plt.text(0, 0, "Revenue")
plt.text(1, 2, "Profit")
plt.text(2, 0, "Customers")
plt.title("Triangle Relationship Chart")
plt.axis('off')
plt.show()
theta = np.linspace(0, 2*np.pi, len(df.head(20)))
r = df['Profit'].head(20)
plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, r, color='gold', linewidth=2)
ax.fill(theta, r, alpha=0.3, color='orange')
plt.title("Polar Profit Chart")
plt.show()
plt.figure(figsize=(8,5))
sizes = df['Customer_Count'] * 5
plt.scatter(
    df['Revenue'],
    df['Profit'],
    s=sizes,
    alpha=0.5,
    color='violet'
)
plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.title("Bubble Chart")
plt.show(
city_profit = df.groupby('City')['Profit'].mean()
plt.figure(figsize=(8,5))
plt.barh(city_profit.index,
         city_profit.values,
         color='skyblue')
plt.title("Profit Pyramid")
plt.xlabel("Profit")
plt.show()
x = np.linspace(0, 20, 500)
y = np.sin(x)
plt.figure(figsize=(10,5))
plt.plot(x, y, linewidth=3, color='blue')
plt.title("Wave Plot")
plt.show()
stages = ['Visitors','Customers','Regulars']
values = [1000, 700, 400]
plt.figure(figsize=(8,5))
plt.barh(stages, values,
         color=['gold','silver','brown'])
plt.title("Customer Funnel")
plt.show()
angles = np.linspace(0, 2*np.pi, 6)
radius = [1,3,1,3,1,3]
ax = plt.subplot(111, projection='polar')
ax.plot(angles, radius,
        color='red',
        linewidth=2)
ax.fill(angles, radius,
        color='pink',
        alpha=0.5)
plt.title("Star Plot")
plt.show()
city_profit = df.groupby('City')['Profit'].mean()
plt.figure(figsize=(8,4))
plt.hlines(y=city_profit.index,
           xmin=0,
           xmax=city_profit.values,
           color='skyblue')
plt.plot(city_profit.values,
         city_profit.index,
         "o",
         color='red')
plt.title("Lollipop Chart")
plt.xlabel("Profit")
plt.show()
plt.figure(figsize=(10,5))
plt.plot(df['Revenue'].head(30),
         linestyle='--',
         marker='o',
         color='purple')
plt.title("Broken Line Revenue")
plt.show()
x = np.arange(20)
y = [(-1)**i * i for i in x]
plt.figure(figsize=(10,5))
plt.plot(x, y,
         linewidth=3,
         color='darkorange')
plt.title("Zigzag Plot")
plt.show()
fuel = df['Fuel_Type'].value_counts()
angles = np.linspace(0, 2*np.pi, len(fuel), endpoint=False)
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
plt.title("Circular Bar Plot")
plt.show()
labels = ['Revenue','Profit','Expenses','Customers','Rating']
stats = [
    df['Revenue'].mean()/100,
    df['Profit'].mean()/100,
    df['Daily_Expense'].mean()/100,
    df['Customer_Count'].mean(),
    df['Customer_Rating'].mean()*20
]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats += stats[:1]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
ax.plot(angles, stats,
        color='blue',
        linewidth=2)
ax.fill(angles, stats,
        color='lightblue',
        alpha=0.4)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Spider Web Chart")
plt.show()
data = np.random.rand(10,10)
plt.figure(figsize=(8,6))
plt.pcolormesh(data)
plt.colorbar()
plt.title("Color Mesh Plot")
plt.show()
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.figure(figsize=(8,6))
plt.contour(X, Y, Z)
plt.title("Contour Plot")
plt.show()
plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.title("Filled Contour Plot")
plt.show()
matrix = df[['Revenue',
             'Profit',
             'Customer_Count',
             'Employee_Count']].head(20)
plt.figure(figsize=(8,6))
plt.imshow(matrix,
           aspect='auto')
plt.colorbar()
plt.title("Matrix Style Visualization")
plt.show()
import matplotlib.pyplot as plt

# KPI Values
total_revenue = df['Revenue'].sum()
total_profit = df['Profit'].sum()
total_customers = df['Customer_Count'].sum()
avg_rating = round(df['Customer_Rating'].mean(),2)

# Create Figure
fig, ax = plt.subplots(figsize=(12,3))

# Remove Axis
ax.axis('off')

# KPI CARD 1
ax.text(0.1, 0.7,
        f"Total Revenue\n₹{total_revenue:,.0f}",
        fontsize=16,
        bbox=dict(facecolor='gold',
                  edgecolor='black',
                  boxstyle='round,pad=1'))

# KPI CARD 2
ax.text(0.35, 0.7,
        f"Total Profit\n₹{total_profit:,.0f}",
        fontsize=16,
        bbox=dict(facecolor='silver',
                  edgecolor='black',
                  boxstyle='round,pad=1'))

# KPI CARD 3
ax.text(0.6, 0.7,
        f"Customers\n{total_customers:,}",
        fontsize=16,
        bbox=dict(facecolor='lightblue',
                  edgecolor='black',
                  boxstyle='round,pad=1'))

# KPI CARD 4
ax.text(0.82, 0.7,
        f"Avg Rating\n{avg_rating}",
        fontsize=16,
        bbox=dict(facecolor='lightgreen',
                  edgecolor='black',
                  boxstyle='round,pad=1'))

plt.title("Fuel Station KPI Dashboard", fontsize=20)

plt.show()
df['Health_Score'] = (
    df['Customer_Rating'] * 20 +
    df['Profit'] / 1000
)
print(df[['Health_Score']].head())
plt.figure(figsize=(8,5))

plt.hist(
    df['Customer_Rating'],
    bins=10,
    color='gold'
)

plt.title("Customer Rating Distribution")

plt.xlabel("Rating")
plt.ylabel("Count")

plt.show()
plt.figure(figsize=(10,5))

plt.plot(df['Revenue'].head(50),
         marker='o',
         linestyle='--',
         color='gold')

plt.title("Revenue Flow Animation Style")
plt.grid(True)

plt.show()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = df[['Revenue', 'Profit']]

kmeans = KMeans(n_clusters=3, random_state=0)

df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8,5))

plt.scatter(
    df['Revenue'],
    df['Profit'],
    c=df['Cluster'],
    cmap='viridis',
    s=80
)

plt.xlabel("Revenue")
plt.ylabel("Profit")
plt.title("AI Customer Clustering")

plt.show()
avg_profit = df['Profit'].mean()
plt.figure(figsize=(6,4))
if avg_profit > 50000:
    plt.bar(
        ['Business Status'],
        [avg_profit],
        color='gold'
    )
    plt.text(
        0,
        avg_profit + 2000,
        'Excellent Performance',
        ha='center',
        fontsize=12
    )
else:
    plt.bar(
        ['Business Status'],
        [avg_profit],
        color='red'
    )
    plt.text(
        0,
        avg_profit + 2000,
        'Needs Improvement',
        ha='center',
        fontsize=12
    )
plt.title("Business Performance Analysis")
plt.ylabel("Average Profit")
plt.show()























