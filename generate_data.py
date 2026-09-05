import numpy as np
import pandas as pd
import os
 
np.random.seed(42)
n_samples = 500
 
area_sqft = np.random.normal(1800, 500, n_samples).clip(400, 5000)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age_years = np.random.randint(0, 50, n_samples)
distance_to_city_km = np.random.uniform(0.5, 30, n_samples)
 
price = (
    area_sqft * 150
    + bedrooms * 5000
    + bathrooms * 3000
    - age_years * 400
    - distance_to_city_km * 800
    + np.random.normal(0, 10000, n_samples)
)
 
df = pd.DataFrame({
    "area_sqft": area_sqft.round(1),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_to_city_km": distance_to_city_km.round(2),
    "price": price.round(2),
})
 
os.makedirs("data", exist_ok=True)
df.to_csv("data/dataset.csv", index=False)
print(f"[INFO] Generated data/dataset.csv with {len(df)} rows.")
 