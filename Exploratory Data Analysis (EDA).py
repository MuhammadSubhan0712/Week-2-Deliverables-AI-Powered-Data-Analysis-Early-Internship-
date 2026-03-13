import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


# loading cleaned dataset:
df = pd.read_csv("Cleaned_Preprocessed_Dataset_Week1.csv")


# Convert datetime columns:
print("++++ Convert datetime columns ++++")
df['Learner SignUp DateTime'] = pd.to_datetime(df['Learner SignUp DateTime'])
df['Apply Date'] = pd.to_datetime(df['Apply Date'], errors='coerce')

print(df.head())
print(df.info())

# ----------------------------------------------------------------------
# SignUp Trends:
print("++++ Signup Growth Over Time ++++")
signup_daily = df.groupby(
    df['Learner SignUp DateTime'].dt.date
).size()

plt.figure(figsize=(12,5))
signup_daily.plot()
plt.title("Daily Signup Trend")
plt.xlabel("Date")
plt.ylabel("Number of Signups")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Signup Seasonality (Month vs Year):
print("++++ Signup Seasonality (Month vs Year) ++++")
seasonality = df.pivot_table(
    index= 'Signup_Month',
    columns='Signup_Year',
    values='Opportunity Id',
    aggfunc='count'
)
plt.figure(figsize=(10,16))
sns.heatmap(seasonality, cmap="Blues")
plt.title("Signup Seasonality Heatmap")
plt.xlabel("Year")
plt.ylabel("Month")
plt.show()

# ----------------------------------------------------------------------
# Signup Spikes and Drops:
print("++++ Signup Spikes and Drops ++++")
top_signup_days = signup_daily.sort_values(ascending=False).head(5)
low_signup_days = signup_daily.sort_values().head(5)

print("Top Signup Days:\n",top_signup_days)
print("\nLowest Signup Days:\n", low_signup_days)

# ----------------------------------------------------------------------
# Completion / Status Analysis:
print("++++ Status Distribution ++++")
plt.figure(figsize=(10,12))
sns.countplot(
    y='Status Description',
    data=df,
    order=df['Status Description'].value_counts().index
)
plt.title("Distribution of Application Status")
plt.xlabel("Count")
plt.ylabel("Status")
plt.show()

# ----------------------------------------------------------------------
# Status Over Time:
print("++++ Status Over Time ++++")
status_time = df.groupby([
    df['Learner SignUp DateTime'].dt.date,
    'Status Description'
]).size().unstack(fill_value=0)

status_time.plot(figsize=(12,6))
plt.title("Status Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Completion Time Behavior:
print("++++ Days to Apply Distribution ++++")
plt.figure(figsize=(8,5))
sns.histplot(df['Days_To_Apply'], bins=30, kde=True)
plt.title("Distribution of Days to Apply")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.show()

# ----------------------------------------------------------------------
# Completion Time Stability (Box Plot):
print("++++ Completion Time Stability (Box Plot) ++++")
plt.figure(figsize=(6,5))
sns.boxplot(y=df['Days_To_Apply'])
plt.title("Days to Apply - Outlier Detection")
plt.ylabel("Days")
plt.show()

# ----------------------------------------------------------------------
# Pattern & Correlations:
print("++++ Signup Volume vs Status Outcome ++++")
status_summary = df.groupby('Status Description').size().reset_index(name='Count')

plt.figure(figsize=(8,10))
sns.barplot(x='Status Description', y='Count', data=status_summary)
plt.title("Application Outcomes Comparison")
plt.xticks(rotation=30)
plt.show()

# ----------------------------------------------------------------------
# Demographic Analysis:
print("++++ Age vs Days to Apply ++++")
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Age',
    y='Days_To_Apply',
    hue='Gender',
    data=df
)
plt.title("Age vs Days to Apply by Gender")
plt.show()

# ----------------------------------------------------------------------
# Country-wise Signup Analysis:
print("++++ Country-wise Signup Analysis ++++")

top_countries = df['Country'].value_counts().head(10)

plt.figure(figsize=(8,10))
top_countries.plot(kind='bar')
plt.title("Top 10 Countries by Signups")
plt.xlabel("Country")
plt.ylabel("Number of Signups")
plt.show()

# ----------------------------------------------------------------------
# Outliers & Anomalies:
print("++++ Long-Tail Users (Very High Days to Apply) ++++")
Q1 = df['Days_To_Apply'].quantile(0.25)
Q3 = df['Days_To_Apply'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[df['Days_To_Apply'] > Q3 + 1.5 * IQR ]

print("Number of long completion cases:", outliers.shape[0])




