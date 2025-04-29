import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
import dataframe_image as dfi

# Define base output directory and subdirectories
base_output_dir = 'output/funnel'
overall_dir = os.path.join(base_output_dir, 'overall')
pinwheel_dir = os.path.join(base_output_dir, 'pinwheel_analysis')
user_type_dir = os.path.join(base_output_dir, 'user_type_analysis')

# Create output directories if they don't exist
os.makedirs(overall_dir, exist_ok=True)
os.makedirs(pinwheel_dir, exist_ok=True)
os.makedirs(user_type_dir, exist_ok=True)

# --- Helper function to save dataframe as PNG (tall format) ---
def save_df_as_png(df, filename):
    """Saves a pandas DataFrame as a PNG image, transposing if wider than tall."""
    df_to_save = df.copy()
    # Transpose if wider than tall for better presentation visuals
    if df_to_save.shape[1] > df_to_save.shape[0]:
        df_to_save = df_to_save.T
    try:
        dfi.export(df_to_save, filename, table_conversion='matplotlib')
        print(f"Table saved to {filename}")
    except Exception as e:
        print(f"Could not save table {filename} as png: {e}. Ensure necessary libraries (e.g., dataframe_image, matplotlib) are installed.")

# Load the data
order_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/order_dataset.csv'
user_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/user_dataset.csv'
orders = pd.read_csv(order_file)
try:
    users = pd.read_csv(user_file)
    existing_user_ids = set(users['user_id'].unique())
except FileNotFoundError:
    print(f"Warning: User dataset file not found at {user_file}. Cannot perform new vs existing analysis.")
    existing_user_ids = set()

# Convert timestamp columns to datetime
timestamp_cols = ['application_start_ts', 'application_complete_ts', 'awaiting_payment_ts', 'repayment_ts']
for col in timestamp_cols:
    orders[col] = pd.to_datetime(orders[col])

# Calculate conversion rates
total_applications = len(orders[orders['application_start_ts'].notna()])
completed_applications = len(orders[orders['application_complete_ts'].notna()])
approved_applications = len(orders[orders['awaiting_payment_ts'].notna()])
deposit_setup = len(orders[orders['repayment_ts'].notna()])

# Calculate Pinwheel metrics at user level
total_users = len(orders['user_id'].unique())
pinwheel_eligible_users = len(orders[orders['user_pinwheel_eligible_at_ap'] == True]['user_id'].unique())
pinwheel_eligibility_rate = pinwheel_eligible_users / total_users

# Get unique users who completed each step
users_with_application = orders[orders['application_start_ts'].notna()]['user_id'].unique()
users_with_completion = orders[orders['application_complete_ts'].notna()]['user_id'].unique()
users_with_approval = orders[orders['awaiting_payment_ts'].notna()]['user_id'].unique()
users_with_deposit = orders[orders['repayment_ts'].notna()]['user_id'].unique()

# Create a DataFrame for chi-square analysis
funnel_steps = ['Application Start', 'Application Complete', 'Approval', 'Deposit Setup']
user_groups = ['Pinwheel Eligible', 'Non-Pinwheel Eligible']

# Prepare data for chi-square analysis
chi_square_results = []
for step, users_at_step in zip(funnel_steps, [users_with_application, users_with_completion, users_with_approval, users_with_deposit]):
    # Get Pinwheel eligible users at this step
    pinwheel_eligible_at_step = set(orders[orders['user_pinwheel_eligible_at_ap'] == True]['user_id'].unique()) & set(users_at_step)
    non_pinwheel_eligible_at_step = set(orders[orders['user_pinwheel_eligible_at_ap'] == False]['user_id'].unique()) & set(users_at_step)
    
    # Calculate conversion rates
    pinwheel_rate = len(pinwheel_eligible_at_step) / pinwheel_eligible_users if pinwheel_eligible_users > 0 else 0
    non_pinwheel_rate = len(non_pinwheel_eligible_at_step) / (total_users - pinwheel_eligible_users) if (total_users - pinwheel_eligible_users) > 0 else 0
    
    # Skip chi-square test for Application Start since everyone has started
    if step == 'Application Start':
        chi_square_results.append({
            'Step': step,
            'Pinwheel Rate': pinwheel_rate,
            'Non-Pinwheel Rate': non_pinwheel_rate,
            'Chi-square Statistic': None,
            'P-value': None,
            'Degrees of Freedom': None,
            'Odds Ratio': None,
            'Significant': None
        })
        continue
    
    # Create contingency table for chi-square test
    pinwheel_success = len(pinwheel_eligible_at_step)
    pinwheel_failure = pinwheel_eligible_users - pinwheel_success
    non_pinwheel_success = len(non_pinwheel_eligible_at_step)
    non_pinwheel_failure = (total_users - pinwheel_eligible_users) - non_pinwheel_success
    
    contingency_table = np.array([
        [pinwheel_success, pinwheel_failure],
        [non_pinwheel_success, non_pinwheel_failure]
    ])
    
    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calculate odds ratio
    odds_ratio = (pinwheel_success * non_pinwheel_failure) / (pinwheel_failure * non_pinwheel_success) if pinwheel_failure * non_pinwheel_success > 0 else 0
    
    chi_square_results.append({
        'Step': step,
        'Pinwheel Rate': pinwheel_rate,
        'Non-Pinwheel Rate': non_pinwheel_rate,
        'Chi-square Statistic': chi2_stat,
        'P-value': p_value,
        'Degrees of Freedom': dof,
        'Odds Ratio': odds_ratio,
        'Significant': p_value < 0.05
    })

# Convert to DataFrame for easier analysis
chi_square_df = pd.DataFrame(chi_square_results)

# Calculate conversion rates
application_completion_rate = completed_applications / total_applications
approval_rate = approved_applications / completed_applications
deposit_setup_rate = deposit_setup / approved_applications if approved_applications > 0 else 0
overall_conversion_rate = deposit_setup / total_applications if total_applications > 0 else 0

# --- User Type Identification ---
all_order_user_ids = set(orders['user_id'].unique())
existing_user_ids_in_orders = existing_user_ids & all_order_user_ids
new_user_ids_in_orders = all_order_user_ids - existing_user_ids

orders['user_type'] = orders['user_id'].apply(lambda x: 'Existing' if x in existing_user_ids_in_orders else 'New')
total_new_users = len(new_user_ids_in_orders)
total_existing_users = len(existing_user_ids_in_orders)

# Create a DataFrame for the funnel
funnel_data = pd.DataFrame({
    'Stage': ['Applications Started', 'Applications Completed', 'Approved', 'Deposit Setup'],
    'Count': [total_applications, completed_applications, approved_applications, deposit_setup],
    'Conversion Rate': [
        1.0,  # 100% at start
        application_completion_rate,
        approval_rate,
        deposit_setup_rate
    ]
})

# Print the results
print("\n=== Funnel Conversion Rates ===")
print(f"\nApplication Completion Rate: {application_completion_rate:.2%}")
print(f"Approval Rate: {approval_rate:.2%}")
print(f"Deposit Setup Rate: {deposit_setup_rate:.2%}")
print(f"Overall Funnel Conversion: {overall_conversion_rate:.2%}")

print("\n=== Pinwheel Impact Analysis ===")
print(f"Total Users: {total_users}")
print(f"Pinwheel Eligible Users: {pinwheel_eligible_users}")
print(f"Pinwheel Eligibility Rate: {pinwheel_eligibility_rate:.2%}")

print("\n=== Chi-square Analysis by Funnel Step ===")
for _, row in chi_square_df.iterrows():
    print(f"\n{row['Step']}:")
    print(f"Pinwheel Rate: {row['Pinwheel Rate']:.2%}")
    print(f"Non-Pinwheel Rate: {row['Non-Pinwheel Rate']:.2%}")
    if row['Step'] != 'Application Start':
        print(f"Chi-square Statistic: {row['Chi-square Statistic']:.4f}")
        print(f"P-value: {row['P-value']:.4f}")
        print(f"Degrees of Freedom: {row['Degrees of Freedom']}")
        print(f"Odds Ratio: {row['Odds Ratio']:.4f}")
        print(f"Significant Difference: {'Yes' if row['Significant'] else 'No'}")

# Create visualization
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create the funnel plot
plt.barh(funnel_data['Stage'], funnel_data['Count'], color='skyblue')
plt.title('Funnel Analysis')
plt.xlabel('Number of Users')
plt.ylabel('Stage')

# Add conversion rate annotations
for i, (count, rate) in enumerate(zip(funnel_data['Count'], funnel_data['Conversion Rate'])):
    if i > 0:
        plt.text(count, i, f'{rate:.1%}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(overall_dir, 'funnel_analysis.png'))
plt.close()

# Create Pinwheel impact visualization
plt.figure(figsize=(12, 6))
x = np.arange(len(funnel_steps))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, chi_square_df['Pinwheel Rate'], width, label='Pinwheel Eligible')
rects2 = ax.bar(x + width/2, chi_square_df['Non-Pinwheel Rate'], width, label='Non-Pinwheel Eligible')

ax.set_ylabel('Conversion Rate')
ax.set_title('Conversion Rates by Funnel Step and Pinwheel Eligibility')
ax.set_xticks(x)
ax.set_xticklabels(funnel_steps)
ax.legend()

# Add significance markers and odds ratios
for i, (_, row) in enumerate(chi_square_df.iterrows()):
    if row['Step'] != 'Application Start' and row['Significant']:
        ax.text(i, max(row['Pinwheel Rate'], row['Non-Pinwheel Rate']) + 0.02, 
                f'* (OR: {row["Odds Ratio"]})', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(pinwheel_dir, 'pinwheel_funnel_impact_original_rates.png'))
plt.close()

# Calculate overall conversion rates for each group (from application start to deposit setup)
pinwheel_deposit_total = len(set(orders[(orders['user_pinwheel_eligible_at_ap'] == True) & (orders['repayment_ts'].notna())]['user_id'].unique()))
pinwheel_total = len(set(orders[orders['user_pinwheel_eligible_at_ap'] == True]['user_id'].unique()))
non_pinwheel_deposit_total = len(set(orders[(orders['user_pinwheel_eligible_at_ap'] == False) & (orders['repayment_ts'].notna())]['user_id'].unique()))
non_pinwheel_total = len(set(orders[orders['user_pinwheel_eligible_at_ap'] == False]['user_id'].unique()))

# Calculate odds ratio for overall conversion
pinwheel_odds = (pinwheel_deposit_total / (pinwheel_total - pinwheel_deposit_total)) if pinwheel_total > 0 and (pinwheel_total - pinwheel_deposit_total) > 0 else np.nan
non_pinwheel_odds = (non_pinwheel_deposit_total / (non_pinwheel_total - non_pinwheel_deposit_total)) if non_pinwheel_total > 0 and (non_pinwheel_total - non_pinwheel_deposit_total) > 0 else np.nan
overall_or_pinwheel = (pinwheel_odds / non_pinwheel_odds) if pinwheel_odds is not np.nan and non_pinwheel_odds is not np.nan and non_pinwheel_odds != 0 else np.nan

# Add this OR to the deposit setup ratio plot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create the bar plot (reuse previous logic for ratios)
pinwheel_deposit_ratio = pinwheel_deposit_total / pinwheel_total if pinwheel_total > 0 else 0
non_pinwheel_deposit_ratio = non_pinwheel_deposit_total / non_pinwheel_total if non_pinwheel_total > 0 else 0

x = np.arange(2)
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x[0], pinwheel_deposit_ratio, width, label='Pinwheel Eligible', color='skyblue')
rects2 = ax.bar(x[1], non_pinwheel_deposit_ratio, width, label='Non-Pinwheel Eligible', color='lightcoral')

ax.set_ylabel('Deposit Setup / Total Application Ratio')
ax.set_title('Deposit Setup Ratio by Pinwheel Eligibility')
ax.set_xticks(x)
ax.set_xticklabels(['Pinwheel Eligible', 'Non-Pinwheel Eligible'])
ax.legend()

# Add overall OR annotation at the top
if not np.isnan(overall_or_pinwheel):
    ax.text(0.5, max(pinwheel_deposit_ratio, non_pinwheel_deposit_ratio) + 0.05,
            f'Overall Conversion OR (Pinwheel vs Non) = {overall_or_pinwheel:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold', transform=ax.transAxes)
else:
    ax.text(0.5, max(pinwheel_deposit_ratio, non_pinwheel_deposit_ratio) + 0.05,
            'Overall Conversion OR (Pinwheel vs Non) = N/A',
            ha='center', va='bottom', fontsize=12, fontweight='bold', transform=ax.transAxes)

# Add percentage labels on top of bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(pinwheel_dir, 'deposit_setup_ratio_analysis_pinwheel.png'))
plt.close()

# --- Corrected Funnel Step Analysis (Pinwheel) ---
# For each step, denominator = users in group who made it to previous step, numerator = users in group who made it to current step

funnel_steps = ['Application Start', 'Application Complete', 'Approval', 'Deposit Setup']
timestamp_cols = ['application_start_ts', 'application_complete_ts', 'awaiting_payment_ts', 'repayment_ts']

# Prepare user sets for each group at each step
user_sets = {
    'Pinwheel Eligible': [],
    'Non-Pinwheel Eligible': []
}

for group, eligible in [('Pinwheel Eligible', True), ('Non-Pinwheel Eligible', False)]:
    prev_users = set(orders[orders['user_pinwheel_eligible_at_ap'] == eligible]['user_id'].unique())
    user_sets[group].append(prev_users)  # Application Start
    for col in timestamp_cols[1:]:
        curr_users = set(orders[(orders['user_pinwheel_eligible_at_ap'] == eligible) & (orders[col].notna())]['user_id'].unique())
        curr_users = prev_users & curr_users  # Only those who made it to previous step and this step
        user_sets[group].append(curr_users)
        prev_users = curr_users

# Now, for each step (except Application Start), calculate conversion rates and ORs
step_labels = funnel_steps[1:]  # Skip Application Start for conversion
pinwheel_counts = [len(user_sets['Pinwheel Eligible'][i]) for i in range(len(funnel_steps))]
non_pinwheel_counts = [len(user_sets['Non-Pinwheel Eligible'][i]) for i in range(len(funnel_steps))]

pinwheel_rates = []
non_pinwheel_rates = []
odds_ratios = []
chi_square_stats = []

for i in range(1, len(funnel_steps)):
    # Numerator: users who made it to this step
    # Denominator: users who made it to previous step
    pinwheel_num = pinwheel_counts[i]
    pinwheel_den = pinwheel_counts[i-1]
    non_pinwheel_num = non_pinwheel_counts[i]
    non_pinwheel_den = non_pinwheel_counts[i-1]
    
    pinwheel_rate = pinwheel_num / pinwheel_den if pinwheel_den > 0 else 0
    non_pinwheel_rate = non_pinwheel_num / non_pinwheel_den if non_pinwheel_den > 0 else 0
    pinwheel_rates.append(pinwheel_rate)
    non_pinwheel_rates.append(non_pinwheel_rate)
    
    # Odds ratio
    pinwheel_fail = pinwheel_den - pinwheel_num
    non_pinwheel_fail = non_pinwheel_den - non_pinwheel_num
    odds_ratio = (pinwheel_num * non_pinwheel_fail) / (pinwheel_fail * non_pinwheel_num) if (pinwheel_fail * non_pinwheel_num) > 0 else np.nan
    odds_ratios.append(odds_ratio)
    
    # Chi-square
    contingency = np.array([
        [pinwheel_num, pinwheel_fail],
        [non_pinwheel_num, non_pinwheel_fail]
    ])
    if np.any(np.array([pinwheel_num, pinwheel_fail, non_pinwheel_num, non_pinwheel_fail]) < 0):
        chi2_stat = np.nan
    else:
        chi2_stat, _, _, _ = stats.chi2_contingency(contingency)
    chi_square_stats.append(chi2_stat)

# Plot corrected funnel step conversion rates
x = np.arange(len(step_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, pinwheel_rates, width, label='Pinwheel Eligible', color='skyblue')
rects2 = ax.bar(x + width/2, non_pinwheel_rates, width, label='Non-Pinwheel Eligible', color='lightcoral')

ax.set_ylabel('Step Conversion Rate')
ax.set_title('Corrected Funnel Step Conversion Rates by Pinwheel Eligibility')
ax.set_xticks(x)
ax.set_xticklabels(step_labels)
ax.legend()

# Add only odds ratio annotations (with * if significant)
for i, (or_val, chi2) in enumerate(zip(odds_ratios, chi_square_stats)):
    if not np.isnan(or_val):
        # Calculate significance (p < 0.05)
        # We already have chi2, so get p-value
        # Reconstruct contingency table for p-value
        pinwheel_num = pinwheel_counts[i+1]
        pinwheel_den = pinwheel_counts[i]
        non_pinwheel_num = non_pinwheel_counts[i+1]
        non_pinwheel_den = non_pinwheel_counts[i]
        pinwheel_fail = pinwheel_den - pinwheel_num
        non_pinwheel_fail = non_pinwheel_den - non_pinwheel_num

        # Handle potential negative counts
        if pinwheel_fail < 0: pinwheel_fail = 0
        if non_pinwheel_fail < 0: non_pinwheel_fail = 0

        contingency = np.array([
            [pinwheel_num, pinwheel_fail],
            [non_pinwheel_num, non_pinwheel_fail]
        ])
        # Ensure all contingency values are non-negative before calculating chi2
        if np.any(contingency < 0):
             p_value = np.nan
             chi2_stat_step = np.nan # Use a different variable name to avoid conflict
        else:
            try:
                chi2_stat_step, p_value, _, _ = stats.chi2_contingency(contingency)
            except ValueError: # Handle cases with zero rows/columns in contingency
                 p_value = np.nan
                 chi2_stat_step = np.nan

        star = '*' if not np.isnan(p_value) and p_value < 0.05 else ''
        ax.text(i, max(pinwheel_rates[i], non_pinwheel_rates[i]) + 0.02,
                f'OR = {or_val:.2f}{star}',
                ha='center', va='bottom', fontsize=10)

# Add percentage labels on top of bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(pinwheel_dir, 'corrected_funnel_step_conversion_rates_pinwheel.png'))
plt.close()

# Save the funnel data to CSV
funnel_data.to_csv(os.path.join(overall_dir, 'funnel_data.csv'), index=False)
save_df_as_png(funnel_data.set_index('Stage'), os.path.join(overall_dir, 'funnel_summary_table.png'))

# Save Pinwheel metrics to CSV
pinwheel_metrics = pd.DataFrame({
    'Metric': ['Total Users', 'Pinwheel Eligible Users', 'Eligibility Rate'],
    'Value': [total_users, pinwheel_eligible_users, pinwheel_eligibility_rate]
})
pinwheel_metrics.to_csv(os.path.join(pinwheel_dir, 'pinwheel_metrics.csv'), index=False)
save_df_as_png(pinwheel_metrics.set_index('Metric'), os.path.join(pinwheel_dir, 'pinwheel_metrics_table.png'))

# Save chi-square results to CSV
chi_square_df.to_csv(os.path.join(pinwheel_dir, 'pinwheel_chi_square_results.csv'), index=False)
save_df_as_png(chi_square_df.set_index('Step'), os.path.join(pinwheel_dir, 'pinwheel_chi_square_table.png'))

# --- Analysis by User Type (New vs Existing) ---

print("\n\n=== Analysis by User Type (New vs Existing) ===")
print(f"Total Users in Orders: {len(all_order_user_ids)}")
print(f"New Users (Not in user_dataset): {total_new_users}")
print(f"Existing Users (In user_dataset): {total_existing_users}")

# Calculate user counts at each step for New vs Existing
user_type_counts = {
    'New': [],
    'Existing': []
}
for col in timestamp_cols:
    new_at_step = len(orders[(orders['user_type'] == 'New') & (orders[col].notna())]['user_id'].unique())
    existing_at_step = len(orders[(orders['user_type'] == 'Existing') & (orders[col].notna())]['user_id'].unique())
    user_type_counts['New'].append(new_at_step)
    user_type_counts['Existing'].append(existing_at_step)

# Calculate overall conversion rates for New vs Existing
new_deposit_total = user_type_counts['New'][-1]
existing_deposit_total = user_type_counts['Existing'][-1]

new_overall_conversion = new_deposit_total / total_new_users if total_new_users > 0 else 0
existing_overall_conversion = existing_deposit_total / total_existing_users if total_existing_users > 0 else 0

print(f"\nOverall Conversion (Deposit Setup / Total Users in Group):")
print(f"New Users: {new_overall_conversion:.2%}")
print(f"Existing Users: {existing_overall_conversion:.2%}")

# Calculate overall OR for New vs Existing
new_odds = (new_deposit_total / (total_new_users - new_deposit_total)) if total_new_users > 0 and (total_new_users - new_deposit_total) > 0 else np.nan
existing_odds = (existing_deposit_total / (total_existing_users - existing_deposit_total)) if total_existing_users > 0 and (total_existing_users - existing_deposit_total) > 0 else np.nan
overall_or_user_type = (new_odds / existing_odds) if new_odds is not np.nan and existing_odds is not np.nan and existing_odds != 0 else np.nan

print(f"Overall Odds Ratio (New vs Existing): {overall_or_user_type:.2f}" if not np.isnan(overall_or_user_type) else "Overall Odds Ratio (New vs Existing): N/A")

# Calculate step-wise conversion rates and ORs for New vs Existing
new_step_rates = []
existing_step_rates = []
user_type_step_odds_ratios = []
user_type_step_chi2 = []
user_type_step_p_values = []

for i in range(1, len(funnel_steps)):
    new_num = user_type_counts['New'][i]
    new_den = user_type_counts['New'][i-1]
    existing_num = user_type_counts['Existing'][i]
    existing_den = user_type_counts['Existing'][i-1]

    new_rate = new_num / new_den if new_den > 0 else 0
    existing_rate = existing_num / existing_den if existing_den > 0 else 0
    new_step_rates.append(new_rate)
    existing_step_rates.append(existing_rate)

    # Odds Ratio & Chi-square
    new_fail = new_den - new_num
    existing_fail = existing_den - existing_num

    # Handle potential negative counts
    if new_fail < 0: new_fail = 0
    if existing_fail < 0: existing_fail = 0

    odds_ratio = (new_num * existing_fail) / (new_fail * existing_num) if (new_fail * existing_num) > 0 else np.nan
    user_type_step_odds_ratios.append(odds_ratio)

    contingency = np.array([
        [new_num, new_fail],
        [existing_num, existing_fail]
    ])

    chi2 = np.nan
    p_value = np.nan
    if np.any(contingency < 0):
        pass # chi2 and p_value remain nan
    else:
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        except ValueError: # Handle cases with zero rows/columns
             pass # chi2 and p_value remain nan
    user_type_step_chi2.append(chi2)
    user_type_step_p_values.append(p_value)


# Create DataFrame for User Type step analysis results
user_type_step_analysis_df = pd.DataFrame({
    'Step': step_labels,
    'New User Rate': new_step_rates,
    'Existing User Rate': existing_step_rates,
    'Odds Ratio (New/Existing)': user_type_step_odds_ratios,
    'Chi-square Statistic': user_type_step_chi2,
    'P-value': user_type_step_p_values,
    'Significant': [p < 0.05 if not np.isnan(p) else False for p in user_type_step_p_values]
})

print("\n=== Step-wise Conversion Analysis (New vs Existing) ===")
print(user_type_step_analysis_df.to_string(index=False, float_format="%.4f"))
user_type_step_analysis_df.to_csv(os.path.join(user_type_dir, 'new_vs_existing_step_analysis.csv'), index=False)
save_df_as_png(user_type_step_analysis_df.set_index('Step'), os.path.join(user_type_dir, 'new_vs_existing_step_analysis_table.png'))


# Plot User Type Overall Conversion Ratio
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
x = np.arange(2)
width = 0.35
fig_ut, ax_ut = plt.subplots(figsize=(10, 6))
rects_new = ax_ut.bar(x[0], new_overall_conversion, width, label='New Users', color='mediumseagreen')
rects_existing = ax_ut.bar(x[1], existing_overall_conversion, width, label='Existing Users', color='tomato')

ax_ut.set_ylabel('Overall Conversion Rate (Deposit Setup / Total)')
ax_ut.set_title('Overall Conversion Rate by User Type')
ax_ut.set_xticks(x)
ax_ut.set_xticklabels(['New Users', 'Existing Users'])
ax_ut.legend()

# Add overall OR annotation
if not np.isnan(overall_or_user_type):
    ax_ut.text(0.5, max(new_overall_conversion, existing_overall_conversion) + 0.05,
            f'Overall Conversion OR (New vs Existing) = {overall_or_user_type:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold', transform=ax_ut.transAxes)
else:
    ax_ut.text(0.5, max(new_overall_conversion, existing_overall_conversion) + 0.05,
            'Overall Conversion OR (New vs Existing) = N/A',
            ha='center', va='bottom', fontsize=12, fontweight='bold', transform=ax_ut.transAxes)


# Add percentage labels
for rect in rects_new + rects_existing:
    height = rect.get_height()
    ax_ut.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(user_type_dir, 'overall_conversion_by_user_type.png'))
plt.close(fig_ut)


# Plot User Type Step Conversion Rates
x_steps = np.arange(len(step_labels))
width = 0.35
fig_ut_step, ax_ut_step = plt.subplots(figsize=(12, 6))
rects1_ut = ax_ut_step.bar(x_steps - width/2, new_step_rates, width, label='New Users', color='mediumseagreen')
rects2_ut = ax_ut_step.bar(x_steps + width/2, existing_step_rates, width, label='Existing Users', color='tomato')

ax_ut_step.set_ylabel('Step Conversion Rate')
ax_ut_step.set_title('Funnel Step Conversion Rates by User Type (New vs Existing)')
ax_ut_step.set_xticks(x_steps)
ax_ut_step.set_xticklabels(step_labels)
ax_ut_step.legend()

# Add odds ratio annotations
for i, row in user_type_step_analysis_df.iterrows():
    or_val = row['Odds Ratio (New/Existing)']
    p_val = row['P-value']
    if not np.isnan(or_val):
        star = '*' if not np.isnan(p_val) and p_val < 0.05 else ''
        ax_ut_step.text(i, max(row['New User Rate'], row['Existing User Rate']) + 0.02,
                f'OR = {or_val:.2f}{star}',
                ha='center', va='bottom', fontsize=10)

# Add percentage labels
for rects in [rects1_ut, rects2_ut]:
    for rect in rects:
        height = rect.get_height()
        ax_ut_step.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(user_type_dir, 'step_conversion_by_user_type.png'))
plt.close(fig_ut_step)


print(f"\nAnalysis complete. Check the '{base_output_dir}' directory and its subdirectories for plots and CSV files.")
