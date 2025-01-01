import pandas as pd
import numpy as np

# Define the number of new rows needed to reach 10,000
current_rows = len(data)
desired_rows = 10000
additional_rows = desired_rows - current_rows

# Generate synthetic data
synthetic_data = pd.DataFrame({
    'item_id': np.random.randint(10000, 20000, additional_rows),
    'location_type': np.random.choice(data['location_type'].unique(), additional_rows),
    'is_discounted': np.random.choice(data['is_discounted'].unique(), additional_rows),
    'category': np.random.choice(data['category'].unique(), additional_rows),
    'is_returnable': np.random.choice(data['is_returnable'].unique(), additional_rows),
    'stock_level': np.random.randint(data['stock_level'].min(), data['stock_level'].max(), additional_rows),
    'brand': ['Zara'] * additional_rows,
    'url': ['https://www.zara.com'] * additional_rows,  # Example placeholder
    'sku': [f"00000000{i}-123-4" for i in range(additional_rows)],
    'name': ['Synthetic Item'] * additional_rows,
    'description': ['Synthetic description'] * additional_rows,
    'price': np.random.uniform(data['price'].min(), data['price'].max(), additional_rows),
    'currency': ['USD'] * additional_rows,
    'scraped_at': np.random.choice(data['scraped_at'].unique(), additional_rows),
    'subcategory': np.random.choice(data['subcategory'].unique(), additional_rows),
    'gender': np.random.choice(data['gender'].unique(), additional_rows),
    'RFID_Tag': np.random.choice(['Yes', 'No'], additional_rows, p=[0.8, 0.2]),
    'actual_stock': np.random.randint(data['stock_level'].min() - 5, data['stock_level'].max() + 5, additional_rows),
    'inaccuracy_rate': np.random.uniform(0, 5, additional_rows),  # Placeholder for a real metric
    'will_stockout': np.random.choice(['Yes', 'No'], additional_rows, p=[0.3, 0.7]),
})

# Combine original data with synthetic data
data_expanded = pd.concat([data, synthetic_data], ignore_index=True)
