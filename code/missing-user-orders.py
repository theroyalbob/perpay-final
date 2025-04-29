import pandas as pd
import numpy as np
from scipy import stats

# Load the data
order_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/order_dataset.csv'
orders = pd.read_csv(order_file)
user_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/user_dataset.csv'
users = pd.read_csv(user_file)

# Find missing user_ids
orders_user_ids = set(orders['user_id'].unique())
users_user_ids = set(users['user_id'].unique())
missing_user_ids = orders_user_ids - users_user_ids
valid_user_ids = orders_user_ids & users_user_ids

# Filter orders for missing and valid user_ids
missing_user_orders = orders[orders['user_id'].isin(missing_user_ids)]
valid_user_orders = orders[orders['user_id'].isin(valid_user_ids)]

# Replace NaN in number_of_payments with 0
missing_user_orders['number_of_payments'] = missing_user_orders['number_of_payments'].fillna(0)
valid_user_orders['number_of_payments'] = valid_user_orders['number_of_payments'].fillna(0)

# Save to CSV
missing_output_file = 'output/missing_user_orders.csv'
valid_output_file = 'output/valid_user_orders.csv'

missing_user_orders.to_csv(missing_output_file, index=False)
valid_user_orders.to_csv(valid_output_file, index=False)

print("\n=== Order Counts ===")
print(f"Found {len(missing_user_orders)} orders from {len(missing_user_ids)} missing users")
print(f"Found {len(valid_user_orders)} orders from {len(valid_user_ids)} valid users")

print("\n=== Summary Statistics Comparison ===")
print("\nMissing User Orders Statistics:")
print(missing_user_orders.describe())
print("\nValid User Orders Statistics:")
print(valid_user_orders.describe())

print("\n=== Categorical Variables Comparison ===")
categorical_columns = ['approval_type', 'cancellation_type']
for col in categorical_columns:
    print(f"\n{col} distribution:")
    print("\nMissing users:")
    print(missing_user_orders[col].value_counts(normalize=True))
    print("\nValid users:")
    print(valid_user_orders[col].value_counts(normalize=True))

print("\n=== ANOVA Analysis ===")
numerical_columns = orders.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    f_stat, p_value = stats.f_oneway(missing_user_orders[col], valid_user_orders[col])
    print(f"\n{col}:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference between groups (p < 0.05)")
    else:
        print("No significant difference between groups (p >= 0.05)")

print(f"\nMissing user orders saved to {missing_output_file}")
print(f"Valid user orders saved to {valid_output_file}") 