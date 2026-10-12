import pandas as pd
import seaborn as sns
import numpy as np

# 1. Load Titanic dataset
df = sns.load_dataset('titanic')

print("Original Dataset:")
print(df.head())
print("\nOriginal Shape:", df.shape)

# 2. Check missing values and duplicates
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Handle missing numerical values using median
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# 5. Handle missing categorical values using mode
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# 6. Clean text columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 7. Convert categorical columns to category data type
categorical_cols = [
    'sex', 'embarked', 'class', 'who', 'deck',
    'embark_town', 'alive'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 8. Detect and cap outliers using IQR
for col in ['age', 'fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower=lower, upper=upper)

# 9. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Shape:", df.shape)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
print("\nDuplicate Rows After Cleaning:", df.duplicated().sum())

# 10. Save cleaned dataset
df.to_csv('titanic_cleaned.csv', index=False)

print("\nCleaned dataset saved as titanic_cleaned.csv")