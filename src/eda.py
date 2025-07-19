# eda.py

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set(style="whitegrid")

def eda():
    data_path="Data/Churn.csv"
    output_dir = "static/eda"
    # Create directory for saving plots
    #os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # -------------------------------
    # 1. Churn Rate
    # -------------------------------
    churn_rate = df['Churn'].value_counts(normalize=True).mul(100).round(2)
    print("🔍 Churn Rate (%):\n", churn_rate)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Churn Distribution')
    plt.savefig(f'{output_dir}/churn_distribution.png')
    plt.close()

    # -------------------------------
    # 2. Categorical Demographics
    # -------------------------------
    for col in ['gender', 'Partner', 'Dependents']:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, hue='Churn', palette='Set1')
        plt.title(f'Churn by {col}')
        plt.savefig(f'{output_dir}/churn_by_{col.lower()}.png')
        plt.close()

    # -------------------------------
    # 3. Tenure Distribution & Churn
    # -------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', bins=30, kde=True, hue='Churn', multiple='stack')
    plt.title('Tenure Distribution by Churn')
    plt.xlabel('Tenure (Months)')
    plt.savefig(f'{output_dir}/tenure_churn_distribution.png')
    plt.close()

    # -------------------------------
    # 4. Contract Type
    # -------------------------------
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set3')
    plt.title('Churn by Contract Type')
    plt.xticks(rotation=15)
    plt.savefig(f'{output_dir}/churn_by_contract.png')
    plt.close()

    # -------------------------------
    # 5. Payment Method
    # -------------------------------
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette='coolwarm')
    plt.title('Churn by Payment Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/churn_by_payment_method.png')
    plt.close()

    print(f"✅ EDA completed. All plots saved to: {output_dir}")
    return

# Example usage (uncomment to run directly)
# eda()
eda()