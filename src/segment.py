import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Load processed data
def segments():
    df = pd.read_csv("Data/processed/data.csv")

# Select features for segmentation
    features = ['tenure', 'MonthlyCharges']
    X = df[features].copy()

# Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Run KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Segment'] = kmeans.fit_predict(X_scaled)

# Add human-readable segment label (optional)
    segment_labels = {
    0: 'Low Tenure + Low Charges',
    1: 'High Tenure + High Charges',
    2: 'Low Tenure + High Charges',
    3: 'High Tenure + Low Charges'
    }
# You can reorder based on mean values of clusters later if needed
# For now, leave numeric

# Optional: flag high-value customers
    df['HighValue'] = df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)

# Save segmented data
    os.makedirs("Data/segmented", exist_ok=True)
    df.to_csv("Data/segmented/segmented_data.csv", index=False)
    return

#print("✅ Customer segmentation completed and saved.")
