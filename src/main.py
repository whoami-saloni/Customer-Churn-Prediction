# src/main.py

import os
from src.preprocess import preprocess_data

from src.segment import segments
from src.train_model import train
from src.evaluate_model import evaluate

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
