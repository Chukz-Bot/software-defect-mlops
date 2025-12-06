import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("SOFTWARE DEFECT PREDICTION\n")

#loading of dataset
print("Loading dataset...")
try:
    df = pd.read_csv('defects.csv')
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
except:
    print(" File not found. Please name your file 'defects.csv'")
    exit()

original_df = df.copy()

#find target colume
target_col = None
for col in df.columns:
    if 'defect' in col.lower():
        target_col = col
        print(f"Target column: '{col}'\n")
        break

if not target_col:
    target_col = df.columns[-1]
    print(f"Using last column as target: '{target_col}'\n")

#sort missing values
print("Check for missing valuee")
missing = df.isnull().sum().sum()

if missing > 0:
    # Fill numeric with median, others with mode
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print(f"Filled {missing} missing values\n")
else:
    print("No missing values\n")

# remove duplicates
print("Checking duplicates...")
duplicates = df.duplicated().sum()

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicates ({duplicates/len(original_df)*100:.1f}%)\n")
else:
    print(" No duplicates\n")

#ENCODE
print("Encoding target variable...")
df[target_col] = df[target_col].astype(str).str.lower()
df[target_col] = df[target_col].map({
    'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0
}).fillna(0).astype(int)

defect_count = df[target_col].sum()
defect_rate = (defect_count / len(df)) * 100
print(f" Binary encoding complete")
print(f"  Non-defective: {len(df) - defect_count:,} ({100-defect_rate:.1f}%)")
print(f"  Defective: {defect_count:,} ({defect_rate:.1f}%)\n")

#scaling features
print("Scaling features...")
feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
print(f"Scaled {len(feature_cols)} features to mean=0, std=1\n")

#clearned dataset
df.to_csv('defects_cleaned.csv', index=False)
print("Saved: defects_cleaned.csv")

#data visulization
print("Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Data Cleaning Results', fontsize=14, fontweight='bold')

# Missing values
axes[0,0].bar(['Before', 'After'], [missing, 0], color=["#0C02D8", "#ECEBF4"])
axes[0,0].set_title('Missing Values')
axes[0,0].set_ylabel('Count')

#Dataset size
axes[0,1].bar(['Before', 'After'], [len(original_df), len(df)], color=["#100d01", "#044174"])
axes[0,1].set_title('Dataset Size')
axes[0,1].set_ylabel('Rows')

#Duplicates
axes[1,0].bar(['Before', 'After'], [duplicates, 0], color=["#e6a5a5", '#51cf66'])
axes[1,0].set_title('Duplicates Removed')
axes[1,0].set_ylabel('Count')

#Target distribution
axes[1,1].bar(['Non-Defective', 'Defective'], 
              [len(df) - defect_count, defect_count],
              color=['#51cf66', '#ff6b6b'])
axes[1,1].set_title('Target Distribution')
axes[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('cleaning_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: cleaning_comparison.png")

# SUMMARY
summary = f"""SUMMARY
{'='*8}

DATASET: defects.csv
Original rows: {len(original_df):,}
Final rows: {len(df):,}
Removed: {len(original_df) - len(df):,} rows

STEPS:
Missing values: {missing} handled
Duplicates: {duplicates} removed
Target encoded: Binary (0/1)
Features scaled: {len(feature_cols)}

TARGET DISTRIBUTION:
Non-defective: {len(df) - defect_count:,} ({100-defect_rate:.1f}%)
Defective: {defect_count:,} ({defect_rate:.1f}%)
Ready for machine learning 
"""
with open('cleaning_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("Saved: cleaning_summary.txt")
print("CLEANING IS COMPLETE.")
print(summary)