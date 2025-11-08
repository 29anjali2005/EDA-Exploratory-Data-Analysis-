import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1.Add Column Names
# Assign the above column names to both datasets appropriately.
print("Step 1: Loading datasets...")
cust = pd.read_csv(r"C:\Users\29anj\OneDrive\Desktop\NetechIndia\PYTHON\customer_details.csv", header=None)
pol = pd.read_csv(r"C:\Users\29anj\OneDrive\Desktop\NetechIndia\PYTHON\customer_policy_details.csv", header=None)

cust.columns = ["customer_id", "Gender", "age", "driving_licence_present",
                "region_code", "previously_insured", "vehicle_age", "vehicle_damage"]

pol.columns = ["customer_id", "annual_premium", "sales_channel_code", "vintage", "response"]
print(cust)
print(pol)
print("Step 1 Completed: Column names assigned successfully!\n")

# 2. Data Quality Check & Cleaning
print("Step 2: Performing Data Cleaning...")

#i. Handling Null Values
print("Checking and filling missing values...")
# Generate Summary Table Showing the count of null values column wise
print(f"DataFrame 1 : Customer Table\n{cust.isnull().sum()}")
print(f"DataFrame 2 : Policy Table\n{pol.isnull().sum()}")

# -Drop Null Values
cust = cust[~(cust['customer_id'].isna())]
pol = pol[~(pol['customer_id'].isna())]
print(cust)
print(pol)

print("Null values handled successfully!\n")
print(f"After Removal of Null Values - Customer_id")
print(f"DataFrame 1 : Customer Table\n{cust.isnull().sum()}")
print(f"DataFrame 2 : Policy Table\n{pol.isnull().sum()}")

# Numeric columns → with mean
# Categorical columns → with mode
numeric_columns_df1 = [col for col in cust.select_dtypes(exclude = 'object')]
categorical_columns_df1 = [col for col in cust.select_dtypes(include = 'object')]
print(f"Customer_d1: {numeric_columns_df1}")
print(f"Policy_d1: {categorical_columns_df1}")

numeric_columns_df2 = [col for col in pol.select_dtypes(exclude = 'object')]
categorical_columns_df2 = [col for col in pol.select_dtypes(include = 'object')]
print(f"Customer_d2: {numeric_columns_df2}")
print(f"Policy_d2:{categorical_columns_df2}")

# Replacement

for col in numeric_columns_df1:
    cust[col] = cust[col].fillna(cust[col].mean())

for col in numeric_columns_df2:
    pol[col] = pol[col].fillna(pol[col].mean())

for col in categorical_columns_df1:
    cust[col] = cust[col].fillna(cust[col].mode().values[0])

for col in categorical_columns_df2:
    pol[col] = pol[col].fillna(pol[col].mode().values[0])

print(f"DataFrame 1 : Customer Table\n{cust.isnull().sum()}")
print(f"DataFrame 2 : Policy Table\n{pol.isnull().sum()}")


# ii Handling Outliers (IQR Method)
print("→ Detecting and treating outliers using IQR method...")

def treat_outliers(df, col):
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df.loc[df[col] < lower, col] = df[col].mean()
        df.loc[df[col] > upper, col] = df[col].mean()

for df in [cust, pol]:
    for col in df.columns:
        treat_outliers(df, col)

print("Outliers treated successfully!\n")

# iii Remove White Spaces
for df in [cust, pol]:
    df_obj = df.select_dtypes(include='object')
    for col in df_obj.columns:
        df[col] = df[col].str.strip()

#iv Case Correction
for df in [cust, pol]:
    df_obj = df.select_dtypes(include='object')
    for col in df_obj.columns:
        df[col] = df[col].str.lower()

# v Convert Nominal Data to Dummies (excluding Gender and vehicle_age)
categorical_cols = cust.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Gender')  # keep Gender as-is
categorical_cols.remove('vehicle_age')  # keep vehicle_age as-is
cust = pd.get_dummies(cust, columns=categorical_cols, drop_first=True)

print("Columns converted to dummies:", categorical_cols)

# vi Drop Duplicates
cust.drop_duplicates(inplace=True)
pol.drop_duplicates(inplace=True)

print("Step 2 Completed: Data cleaned successfully!\n")


# 3. Create Master Table
print("Step 3: Creating Master Table...")
master = pd.merge(cust, pol, on='customer_id', how='inner')

# Combine gender from dummies into single column (if needed)
if "vehicle_damage_yes" in master.columns and "vehicle_damage_no" in master.columns:
    master["vehicle_damage"] = master["vehicle_damage_yes"].apply(lambda x: "yes" if x == 1 else "no")
    master.drop(columns=["vehicle_damage_yes", "vehicle_damage_no"], inplace=True)
master.to_csv("cleaned_master_table.csv", index=False)
print(f"Final master table shape: {master.shape}")
print(f"All Column : {master.columns}")
print("Step 3 Completed: Master table created!\n")
print(master)


# 4. Generate Insights
print("Step 4: Generating Insights...")
# i. What is the gender-wise average annual premium?
valid_genders = ["male", "female"]
filtered_master = master[master["Gender"].isin(valid_genders)]
gender_avg_premium = filtered_master.groupby("Gender")["annual_premium"].mean()
print(gender_avg_premium)

# ii. What is the age-wise average annual premium?
age_avg_premium = master.groupby("age")["annual_premium"].mean()

# iii. Is your data balanced between genders?
vehicle_age_avg_premium = master.groupby("vehicle_age")["annual_premium"].mean()

# iv. What is the vehicle age-wise average annual premium?
gender_balance = master["Gender"].value_counts()

print("Step 4 Completed: Insights generated!\n")

# 5. Analyze Relationship
print("Step 5: Correlation Analysis...")
corr = master['age'].corr(master['annual_premium'])
print(f"Correlation between Age and Annual Premium: {corr:.2f}")
if corr < -0.5:
    print("Interpretation: Strong Negative Relationship")
elif corr > 0.5:
    print("Interpretation: Strong Positive Relationship")
else:
    print("Interpretation: No Significant Relationship")
print(corr)
print("Step 5 Completed: Correlation analysis done!\n")

# 6. Visualizations
print("Step 6: Generating Visualizations...")

if not os.path.exists("visuals"):
    os.makedirs("visuals")

# # Gender-wise Premium
gender_avg_premium.plot(kind='bar', color=['blue', 'pink'])
plt.title("Gender-wise Average Annual Premium")
plt.xlabel("Gender")
plt.ylabel("Average Premium (Rs)")
plt.show()
plt.close()

# Age-wise Premium
age_avg_premium.plot(kind='line', color='purple')
plt.title("Age-wise Average Annual Premium")
plt.xlabel("Age")
plt.ylabel("Average Premium (Rs)")
plt.show()
plt.close()

# Vehicle Age vs Premium
vehicle_age_avg_premium.plot(kind='bar', color='green')
plt.title("Vehicle Age vs Average Annual Premium")
plt.xlabel("Vehicle Age")
plt.ylabel("Average Premium (Rs)")
plt.show()
plt.close()

# Correlation Heatmap
corr_matrix = master.corr(numeric_only=True)
plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.colorbar()
plt.show()
plt.close()

print("Step 6 Completed: Visualizations saved in 'visuals/' folder!\n")

# 7. Final Dataset Summary
print("Final Summary Overview:")
summary = pd.DataFrame({
    "Dataset": ["cust", "pol", "master"],
    "Rows": [cust.shape[0], pol.shape[0], master.shape[0]],
    "Columns": [cust.shape[1], pol.shape[1], master.shape[1]],
    "Null Values": [cust.isnull().sum().sum(), pol.isnull().sum().sum(), master.isnull().sum().sum()]
})
print(summary)

# 8. Final Completion Message
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("Cleaned data saved as: cleaned_master_table.csv")
print("Visuals available in: visuals/ folder")
print("Graphs displayed successfully.")

