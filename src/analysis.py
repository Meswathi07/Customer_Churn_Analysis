# ==========================================================
# KPI-Driven Revenue Leakage & Customer Retention Analysis
# Industry-Grade Professional Pipeline (Ultimate Version)
# ==========================================================

import pandas as pd
import os

# ----------------------------------------------------------
# 1️⃣ LOAD DATA
# ----------------------------------------------------------

print("\n==============================")
print("LOADING DATASET")
print("==============================\n")

df = pd.read_csv("../data/telecom_churn.csv")

print("Dataset loaded successfully!\n")


# ----------------------------------------------------------
# 2️⃣ COMPLETE DATA INSPECTION
# ----------------------------------------------------------

print("\n==============================")
print("DATA INSPECTION")
print("==============================\n")

print("First 5 Rows:\n")
print(df.head())

print("\nLast 5 Rows:\n")
print(df.tail())

print("\nDataset Info:\n")
df.info()

print("\nMissing Values Per Column:\n")
print(df.isnull().sum())

duplicate_count = df['customerID'].duplicated().sum()
print(f"\nDuplicate Customer IDs: {duplicate_count}")

print("\nStatistical Summary:\n")
print(df.describe(include="all"))

print("\nRevenue Range Check:")
print("Minimum MonthlyCharges:", df['MonthlyCharges'].min())
print("Maximum MonthlyCharges:", df['MonthlyCharges'].max())

print("\nTenure Range Check:")
print(df['tenure'].describe())

print("\n==============================\n")


# ----------------------------------------------------------
# 3️⃣ DATA CLEANING
# ----------------------------------------------------------

print("\n==============================")
print("DATA CLEANING")
print("==============================\n")

initial_rows = df.shape[0]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(inplace=True)
after_null_removal = df.shape[0]

df.drop_duplicates(inplace=True)
after_duplicate_removal = df.shape[0]

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df.rename(columns={'MonthlyCharges': 'monthly_revenue'}, inplace=True)

print(f"Rows before cleaning: {initial_rows}")
print(f"Rows after null removal: {after_null_removal}")
print(f"Rows after duplicate removal: {after_duplicate_removal}")
print(f"Total rows removed: {initial_rows - after_duplicate_removal}")

print("\nData Cleaning Completed!\n")


# ----------------------------------------------------------
# 4️⃣ SEGMENT CONFIRMATION ANALYSIS
# ----------------------------------------------------------

print("\n==============================")
print("SEGMENT CONFIRMATION ANALYSIS")
print("==============================\n")

print("Churn Rate by Contract Type:\n")
print(df.groupby('Contract')['Churn'].mean())

print("\nChurn Rate by Payment Method:\n")
print(df.groupby('PaymentMethod')['Churn'].mean())

print("\nChurn Rate by Senior Citizen:\n")
print(df.groupby('SeniorCitizen')['Churn'].mean())

print("\nChurn Rate for Tenure < 12 Months:")
print(df[df['tenure'] < 12]['Churn'].mean())

print("\nChurn Rate for Tenure >= 12 Months:")
print(df[df['tenure'] >= 12]['Churn'].mean())

print("\n==============================\n")


# ----------------------------------------------------------
# 5️⃣ BUSINESS FEATURE ENGINEERING
# ----------------------------------------------------------

print("\n==============================")
print("BUSINESS FEATURE ENGINEERING")
print("==============================\n")

# Revenue threshold (75th percentile)
revenue_75 = df['monthly_revenue'].quantile(0.75)

df['high_risk'] = (
    (df['Contract'] == 'Month-to-month') |
    (df['tenure'] < 12)
)

df['high_value'] = df['monthly_revenue'] > revenue_75

df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years']
)

print("Business Columns Added Successfully!\n")


# ----------------------------------------------------------
# 6️⃣ KPI CALCULATION LAYER
# ----------------------------------------------------------

print("\n==============================")
print("KPI CALCULATIONS")
print("==============================\n")

total_customers = df.shape[0]
total_revenue = df['monthly_revenue'].sum()
churn_rate = df['Churn'].mean()
retention_rate = 1 - churn_rate
revenue_lost = df[df['Churn'] == 1]['monthly_revenue'].sum()
revenue_at_risk = df[df['high_risk']]['monthly_revenue'].sum()
risk_percentage = (revenue_at_risk / total_revenue) * 100
avg_revenue_churned = df[df['Churn'] == 1]['monthly_revenue'].mean()

print(f"Total Customers: {total_customers}")
print(f"Total Monthly Revenue: {round(total_revenue,2)}")
print(f"Churn Rate: {round(churn_rate*100,2)}%")
print(f"Retention Rate: {round(retention_rate*100,2)}%")
print(f"Revenue Lost (Churned Customers): {round(revenue_lost,2)}")
print(f"Revenue At Risk (High Risk Customers): {round(revenue_at_risk,2)}")
print(f"Percentage of Revenue At Risk: {round(risk_percentage,2)}%")
print(f"Average Revenue per Churned Customer: {round(avg_revenue_churned,2)}")

print("\nRevenue Lost by Contract Type:\n")
print(df[df['Churn']==1].groupby('Contract')['monthly_revenue'].sum())

print("\nRevenue Lost by Tenure Group:\n")
print(df[df['Churn']==1].groupby('tenure_group', observed=False)['monthly_revenue'].sum())

print("\n==============================\n")

# ----------------------------------------------------------
# 9️⃣ PREMIUM RISK SEGMENTATION (Advanced Layer)
# ----------------------------------------------------------

print("\n==============================")
print("PREMIUM RISK SEGMENTATION")
print("==============================\n")

# Create premium_at_risk column
df['premium_at_risk'] = df['high_value'] & df['high_risk']

# Count premium at risk customers
premium_risk_count = df['premium_at_risk'].sum()

# Revenue from premium at risk customers
premium_risk_revenue = df[df['premium_at_risk']]['monthly_revenue'].sum()

# Percentage of total revenue exposed
premium_risk_percentage = (premium_risk_revenue / total_revenue) * 100

print(f"Premium Customers At Risk (Count): {premium_risk_count}")
print(f"Revenue From Premium At Risk Customers: {round(premium_risk_revenue,2)}")
print(f"Percentage of Total Revenue Exposed (Premium Risk): {round(premium_risk_percentage,2)}%")

print("\n==============================\n")

# ----------------------------------------------------------
# 7️⃣ FINAL VALIDATION
# ----------------------------------------------------------

print("Final Dataset Info:\n")
df.info()

print("\nFinal Dataset Shape:")
print(df.shape)

print("\n==============================\n")

print("\n==============================")
print("INSIGHT SUMMARY")
print("==============================\n")

print(f"Overall churn rate is {round(churn_rate*100,2)}%, meaning approximately 1 in 4 customers are leaving.")

print("Month-to-month contracts show significantly higher churn compared to long-term contracts.")

print("Customers in the first 12 months show the highest instability and churn risk.")

print(f"Total revenue lost due to churn is {round(revenue_lost,2)}, indicating substantial financial impact.")

print(f"Approximately {round(risk_percentage,2)}% of total revenue is currently at risk from high-risk customers.")

print("Revenue leakage is primarily driven by short-tenure and flexible-contract customers.")

# ----------------------------------------------------------
# 8️⃣ SAVE PROCESSED DATA
# ----------------------------------------------------------

output_path = "../processed_data"

if not os.path.exists(output_path):
    os.makedirs(output_path)

df.to_csv(f"{output_path}/cleaned_churn_data.csv", index=False)

print("\nProcessed dataset saved successfully in processed_data folder!\n")