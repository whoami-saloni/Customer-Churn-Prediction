# src/main.py

import os
from preprocess import preprocess_data
from eda import perform_eda
from segment import segments
from train_model import train
from evaluate_model import evaluate

def run_pipeline():
    print("\n🔧 Step 1: Preprocessing data...")
    preprocess_data()

    #print("\n📊 Step 2: Performing Exploratory Data Analysis...")
    #perform_eda()

    print("\n📦 Step 3: Segmenting Customers...")
    segments()

    print("\n🤖 Step 4: Training ML Model...")
    train()

    print("\n📈 Step 5: Evaluating Model...")
    evaluate()

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
