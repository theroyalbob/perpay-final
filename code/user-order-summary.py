import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as tbl # Import for saving tables
import seaborn as sns
from datetime import datetime
import os # Keep os import

# --- Helper Function to Save DataFrame/Series as PNG ---
def save_df_as_png(df, path, title=""):
    """Saves a pandas DataFrame or Series as a PNG image.
    Transposes the DataFrame if it is wider than it is tall.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame() # Convert Series to DataFrame for table plotting

    # --- Transpose if wider than tall ---
    if df.shape[1] > df.shape[0]:
        df = df.T
    # ------------------------------------

    fig, ax = plt.subplots(figsize=(max(8, df.shape[1] * 1.5), max(4, df.shape[0] * 0.5))) # Adjust size dynamically
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    the_table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         loc='center',
                         cellLoc='center') # Center text in cells

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10) # Adjust font size as needed
    the_table.scale(1.2, 1.2) # Scale table size

    # Add title if provided
    if title:
        plt.title(title, fontsize=14, weight='bold')

    plt.tight_layout(pad=1.0) # Add padding
    plt.savefig(path, dpi=150, bbox_inches='tight') # Increase DPI for better quality
    plt.close(fig) # Close the figure to free memory

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_palette("husl")

# Load the data
order_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/order_dataset.csv'
orders = pd.read_csv(order_file)
user_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/user_dataset.csv'
users = pd.read_csv(user_file)

# Convert timestamp columns to datetime
users['signup_dt'] = pd.to_datetime(users['signup_dt'])
users['last_login'] = pd.to_datetime(users['last_login'])
orders['application_start_ts'] = pd.to_datetime(orders['application_start_ts'])
orders['application_complete_ts'] = pd.to_datetime(orders['application_complete_ts'])
orders['awaiting_payment_ts'] = pd.to_datetime(orders['awaiting_payment_ts'])
orders['repayment_ts'] = pd.to_datetime(orders['repayment_ts'])

# --- Create output subdirectories ---
output_base = 'output'
output_dirs = {
    'user': os.path.join(output_base, 'user_reports'),
    'order': os.path.join(output_base, 'order_reports'),
    'pipeline': os.path.join(output_base, 'pipeline_analysis'),
    'spending': os.path.join(output_base, 'user_spending')
}
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# --- Calculate Proportion of Order User IDs Present in Users Dataset ---
order_user_ids = set(orders['user_id'].unique())
user_user_ids = set(users['user_id'].unique())

common_user_ids = order_user_ids.intersection(user_user_ids)
proportion_common = len(common_user_ids) / len(order_user_ids) if len(order_user_ids) > 0 else 0

print(f"\nProportion of user_ids in orders that are also in users: {proportion_common:.2%}")
print(f"Total unique user_ids in orders: {len(order_user_ids)}")
print(f"Total unique user_ids in users: {len(user_user_ids)}")
print(f"Number of common user_ids: {len(common_user_ids)}")

# --- User Segmentation ---
existing_user_ids = set(users['user_id'])
orders['user_type'] = orders['user_id'].apply(lambda x: 'Existing' if x in existing_user_ids else 'New')
print("\nUser Segmentation:")
user_type_counts = orders['user_type'].value_counts()
print(user_type_counts)
# Check if the counts match the proportion calculation
if len(order_user_ids) > 0:
    proportion_existing_from_segmentation = user_type_counts.get('Existing', 0) / orders['user_id'].nunique()
    print(f"Proportion of 'Existing' users from segmentation: {proportion_existing_from_segmentation:.2%}") # This should match the previous calculation if logic is consistent
else:
    print("No unique users in orders to calculate segmentation proportion.")

save_df_as_png(user_type_counts, os.path.join(output_base, 'user_type_distribution.png'), title='Distribution Of User Types In Orders')

# User Report (Primarily based on users dataset, no segmentation needed here)
print("\n=== User Report ===")
user_report_dir = output_dirs['user']

# Categorical Analysis
print("\nTop 10 Employers:")
top_employers = users['company_name'].value_counts().head(10)
print(top_employers)
save_df_as_png(top_employers, os.path.join(user_report_dir, 'top_employers_table.png'), title='Top 10 Employers')

# Plot for top employers
plt.figure(figsize=(12, 6))
top_employers.plot(kind='bar')
plt.title('Top 10 Employers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(user_report_dir, 'top_employers_chart.png'))
plt.close() # Close plot

# Boolean indicators analysis
print("\nBoolean Indicators:")
for col in ['valid_phone_ind', 'was_referred_ind']:
    counts = users[col].value_counts()
    percentages = users[col].value_counts(normalize=True) * 100
    print(f"\n{col}:")
    print(counts)
    print(f"Percentages:\n{percentages}")
    # Save tables
    save_df_as_png(counts, os.path.join(user_report_dir, f'{col}_counts.png'), title=f'{col.replace("_", " ").title()} Counts')
    save_df_as_png(percentages.round(2), os.path.join(user_report_dir, f'{col}_percentages.png'), title=f'{col.replace("_", " ").title()} Percentages')

# Numeric Analysis - spending_limit_est
print("\nSpending Limit Statistics:")
spending_limit_stats = users['spending_limit_est'].describe()
print(spending_limit_stats)
save_df_as_png(spending_limit_stats.round(2), os.path.join(user_report_dir, 'spending_limit_stats.png'), title='Spending Limit Statistics')


# Create distribution plots for spending_limit_est
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.histplot(users['spending_limit_est'], bins=50)
plt.title('Distribution Of Spending Limit')
plt.subplot(1, 2, 2)
sns.boxplot(y=users['spending_limit_est'])
plt.title('Boxplot Of Spending Limit')
plt.tight_layout()
plt.savefig(os.path.join(user_report_dir, 'spending_limit_distribution.png'))
plt.close() # Close plot

# Temporal Analysis (Signup/Login)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.histplot(users['signup_dt'].dropna(), bins=50) # Drop NA for plotting
plt.title('Distribution Of Signup Dates')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.histplot(users['last_login'].dropna(), bins=50) # Drop NA for plotting
plt.title('Distribution Of Last Login Dates')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(user_report_dir, 'user_temporal_distributions.png'))
plt.close() # Close plot

# Calculate recency
today = pd.Timestamp.now()
users['days_since_last_login'] = (today - users['last_login']).dt.days
print("\nDays since last login statistics:")
last_login_stats = users['days_since_last_login'].describe()
print(last_login_stats)
save_df_as_png(last_login_stats.round(2), os.path.join(user_report_dir, 'days_since_last_login_stats.png'), title='Days Since Last Login Statistics')

# --- Order Report (Segmented by User Type) ---
print("\n=== Order Report (Segmented) ===")
order_report_dir = output_dirs['order']

# Numeric Analysis (Segmented)
print("\nAmount Statistics by User Type:")
amount_stats = orders.groupby('user_type')['amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(amount_stats)
save_df_as_png(amount_stats.round(2), os.path.join(order_report_dir, 'amount_stats_by_user_type.png'), title='Amount Statistics By User Type')

print("\nNumber of Payments Statistics by User Type:")
payments_stats = orders.groupby('user_type')['number_of_payments'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(payments_stats)
save_df_as_png(payments_stats.round(2), os.path.join(order_report_dir, 'num_payments_stats_by_user_type.png'), title='Number Of Payments Statistics By User Type')


# Create distribution plots for numeric variables (Segmented)
plt.figure(figsize=(15, 10))

# --- Amount Plot ---
ax1 = plt.subplot(2, 2, 1)
sns.histplot(data=orders, x='amount', hue='user_type', bins=50, kde=True, element="step", ax=ax1)
ax1.set_title('Distribution Of Order Amount By User Type')

ax2 = plt.subplot(2, 2, 2)
sns.boxplot(data=orders, y='amount', x='user_type', ax=ax2) # Use x for hue separation
ax2.set_title('Boxplot Of Order Amount By User Type')
# Add median lines for Amount Boxplot (median is already part of boxplot)
# It seems user wanted median lines on violin plots, which are below.
# Let's adjust the Amount plot to be Violin instead of Boxplot to add the line

plt.close() # Close the previous figure setup

# Rerun figure setup for Amount/Num Payments
fig_order_numeric, axes_order_numeric = plt.subplots(2, 2, figsize=(15, 10))

# Amount Histogram
categories_amount = sorted(orders['user_type'].dropna().unique())
sns.histplot(data=orders, x='amount', hue='user_type', bins=50, kde=True, element="step", ax=axes_order_numeric[0, 0], hue_order=categories_amount)
axes_order_numeric[0, 0].set_title('Distribution Of Order Amount By User Type')

# Amount Violin plot + Median Lines
sns.violinplot(data=orders, y='amount', x='user_type', ax=axes_order_numeric[0, 1], order=categories_amount) # Explicit order
axes_order_numeric[0, 1].set_title('Violin Plot Of Order Amount By User Type')
# Add median lines
palette_amount = sns.color_palette("husl", len(categories_amount))
color_map_amount = dict(zip(categories_amount, palette_amount))
medians_amount = orders.groupby('user_type')['amount'].median()
print("\n--- Plotting Medians for Order Amount ---") # DEBUG
for cat in categories_amount:
    if cat in medians_amount.index:
        median_val = medians_amount[cat]
        line_color = color_map_amount[cat]
        # --- DEBUG PRINT --- 
        print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
        # -------------------
        axes_order_numeric[0, 1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)

# Number of Payments Histogram
categories_payments = sorted(orders['user_type'].dropna().unique()) # Define categories here
sns.histplot(data=orders, x='number_of_payments', hue='user_type', bins=30, kde=True, element="step", binwidth=1, ax=axes_order_numeric[1, 0], hue_order=categories_payments) # Explicit hue order
axes_order_numeric[1, 0].set_title('Distribution Of Number Of Payments By User Type')

# Number of Payments Violin plot + Median Lines
sns.violinplot(data=orders, y='number_of_payments', x='user_type', ax=axes_order_numeric[1, 1], order=categories_payments) # Explicit order
axes_order_numeric[1, 1].set_title('Violin Plot Of Number Of Payments By User Type')
# Add median lines
# categories_payments = sorted(orders['user_type'].dropna().unique()) # Now defined above
palette_payments = sns.color_palette("husl", len(categories_payments))
color_map_payments = dict(zip(categories_payments, palette_payments))
medians_payments = orders.groupby('user_type')['number_of_payments'].median()

print("\n--- Plotting Medians for Number of Payments ---") # DEBUG
for cat in categories_payments:
    if cat in medians_payments.index:
        median_val = medians_payments[cat]
        line_color = color_map_payments[cat]
        # --- DEBUG PRINT --- 
        print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
        # -------------------
        axes_order_numeric[1, 1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)


plt.tight_layout()
plt.savefig(os.path.join(order_report_dir, 'order_numeric_distributions_segmented.png'))
plt.close(fig_order_numeric)

# Categorical Analysis (Segmented)
categorical_cols = ['approval_type', 'cancellation_type', 'user_pinwheel_eligible_at_ap', 'risk_tier_at_uw']
for col in categorical_cols:
    print(f"\n{col} distribution by User Type:")
    # Calculate counts and percentages
    counts_by_type = orders.groupby('user_type')[col].value_counts().unstack(fill_value=0)
    percentages_by_type = counts_by_type.apply(lambda x: x*100 / x.sum(), axis=1).round(2) # Percentage within each user type

    print("Counts:")
    print(counts_by_type)
    save_df_as_png(counts_by_type, os.path.join(order_report_dir, f'{col}_counts_by_user_type.png'), title=f'{col.replace("_", " ").title()} Counts By User Type')

    print("\nPercentages:")
    print(percentages_by_type)
    save_df_as_png(percentages_by_type, os.path.join(order_report_dir, f'{col}_percentages_by_user_type.png'), title=f'{col.replace("_", " ").title()} Percentages By User Type')

    # Plotting (stacked bar chart for comparison)
    fig, ax = plt.subplots(figsize=(12, 7))
    percentages_by_type.plot(kind='bar', stacked=True, ax=ax)
    plt.title(f'Distribution Of {col.replace("_", " ").title()} By User Type (Percentages)')
    plt.xlabel('User Type')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title=col.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(os.path.join(order_report_dir, f'{col}_distribution_segmented.png'))
    plt.close(fig)

# Temporal Analysis - Application Volume (Segmented)
orders['application_date'] = orders['application_start_ts'].dt.date
daily_volume_segmented = orders.groupby(['application_date', 'user_type']).size().unstack(fill_value=0)

plt.figure(figsize=(15, 6))
daily_volume_segmented.plot(kind='line', marker='.') # Use line plot for trends
plt.title('Daily Application Volume By User Type')
plt.xlabel('Date')
plt.ylabel('Number of Applications')
plt.xticks(rotation=45)
plt.legend(title='User Type')
plt.tight_layout()
plt.savefig(os.path.join(order_report_dir, 'daily_application_volume_segmented.png'))
plt.close() # Close plot

# Weekly volume (Segmented)
orders['week'] = orders['application_start_ts'].dt.isocalendar().week
weekly_volume_segmented = orders.groupby(['week', 'user_type']).size().unstack(fill_value=0)

plt.figure(figsize=(15, 6))
weekly_volume_segmented.plot(kind='line', marker='.') # Use line plot
plt.title('Weekly Application Volume By User Type')
plt.xlabel('Week Number')
plt.ylabel('Number of Applications')
plt.legend(title='User Type')
plt.tight_layout()
plt.savefig(os.path.join(order_report_dir, 'weekly_application_volume_segmented.png'))
plt.close() # Close plot


# --- Pipeline Sequencing Analysis (Segmented) ---
print("\n=== Pipeline Sequencing Analysis (Segmented) ===")
pipeline_dir = output_dirs['pipeline']

# Chronological integrity check (remains the same, violations are per order)
print("\nChecking chronological integrity of timestamps...")
violations = pd.DataFrame()
timestamp_pairs = [
    ('application_start_ts', 'application_complete_ts'),
    ('application_complete_ts', 'awaiting_payment_ts'),
    ('awaiting_payment_ts', 'repayment_ts')
]
for start_col, end_col in timestamp_pairs:
    mask = orders[start_col] > orders[end_col]
    if mask.any():
        num_violations = mask.sum()
        print(f"\nFound {num_violations} violations where {start_col} > {end_col}")
        # Include user_type in violation report
        violations = pd.concat([
            violations,
            orders.loc[mask, ['order_id', 'user_id', 'user_type', start_col, end_col]]
        ])
if len(violations) > 0:
    print("\nSample of violations:")
    print(violations.head())
    violations_path = os.path.join(pipeline_dir, 'timestamp_violations.csv')
    violations.to_csv(violations_path, index=False)
    print(f"Full violation list saved to: {violations_path}")
    # Save a summary table as PNG
    violation_summary = violations.groupby('user_type').size().reset_index(name='Violation Count')
    save_df_as_png(violation_summary, os.path.join(pipeline_dir, 'timestamp_violation_summary.png'), title='Timestamp Violations By User Type')
else:
    print("No chronological violations found!")


# Calculate stage durations (already done)
orders['application_duration'] = (orders['application_complete_ts'] - orders['application_start_ts']).dt.total_seconds() / 60 # Convert to minutes
orders['approval_duration'] = (orders['awaiting_payment_ts'] - orders['application_complete_ts']).dt.total_seconds() / 60 # Convert to minutes
orders['time_to_repayment_start'] = (orders['repayment_ts'] - orders['awaiting_payment_ts']).dt.total_seconds() / (3600 * 24)  # Convert to days

# Analyze durations by user type
duration_cols = ['application_duration', 'approval_duration', 'time_to_repayment_start'] # Rename here
duration_units = ['minutes', 'minutes', 'days'] # Update unit here
percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

print("\nStage Durations Analysis by User Type:")
for col, unit in zip(duration_cols, duration_units):
    print(f"\n{col.replace('_', ' ').title()} Statistics ({unit}) by User Type:")
    # Filter out potential negative durations from violations before describe
    # Also handle potential NaN values if the column doesn't exist or has issues
    valid_durations = orders.loc[orders[col].notna() & (orders[col] >= 0)] if col in orders.columns else pd.DataFrame()
    if not valid_durations.empty:
        stats = valid_durations.groupby('user_type')[col].describe(percentiles=percentiles)
        print(stats)
        save_df_as_png(stats.round(2), os.path.join(pipeline_dir, f'{col}_stats_by_user_type.png'), title=f'{col.replace("_", " ").title()} Stats ({unit}) By User Type') # Unit in title is updated automatically

        # Create detailed distribution plots (Segmented)
        fig_dur, axes_dur = plt.subplots(1, 2, figsize=(15, 5))

        # Define categories, palette, colormap, and medians needed for plot order and lines
        categories_dur = sorted(valid_durations['user_type'].dropna().unique())
        palette_dur = sns.color_palette("husl", len(categories_dur))
        color_map_dur = dict(zip(categories_dur, palette_dur))
        medians_dur = valid_durations.groupby('user_type')[col].median()

        # Histogram
        sns.histplot(data=valid_durations, x=col, hue='user_type', bins=50, kde=True, element="step", ax=axes_dur[0], hue_order=categories_dur) # Explicit hue order
        axes_dur[0].set_title(f'Distribution Of {col.replace("_", " ").title()} ({unit}) By User Type') # Update title format
        axes_dur[0].set_xlabel(f'Duration ({unit})') # Unit in label is updated automatically
        axes_dur[0].grid(True, axis='x')
        if col == 'time_to_repayment_start': # Check against new name
            axes_dur[0].axvline(x=7, color='r', alpha=0.3, linewidth=2) # Change reference line to 7

        # Violin plot
        sns.violinplot(data=valid_durations, y=col, x='user_type', ax=axes_dur[1], order=categories_dur) # Explicit order
        axes_dur[1].set_title(f'Violin Plot Of {col.replace("_", " ").title()} ({unit}) By User Type') # Update title format
        axes_dur[1].set_ylabel(f'Duration ({unit})') # Unit in label is updated automatically
        axes_dur[1].set_xlabel('User Type') # Add x-axis label
        axes_dur[1].grid(True, axis='y')

        # Add median lines for Duration Violin Plot
        print(f"\n--- Plotting Medians for {col.replace('_', ' ').title()} ---") # DEBUG
        for cat in categories_dur:
            if cat in medians_dur.index:
                median_val = medians_dur[cat]
                line_color = color_map_dur[cat]
                # --- DEBUG PRINT --- 
                print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
                # -------------------
                axes_dur[1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)

        if col == 'time_to_repayment_start': # Check against new name
            axes_dur[1].axhline(y=7, color='r', alpha=0.3, linewidth=2) # Change reference line to 7

        plt.tight_layout()
        plt.savefig(os.path.join(pipeline_dir, f'{col}_detailed_segmented.png'))
        plt.close(fig_dur) # Close plot
    else:
        print(f"Could not generate plots for {col} due to missing or invalid data.")

# --- Repayment Timing Analysis (Segmented) ---
print("\n--- Repayment Timing Analysis (Cumulative, Segmented) ---")
orders_entered_repayment = orders[orders['repayment_ts'].notna() & orders['time_to_repayment_start'].notna() & (orders['time_to_repayment_start'] >= 0)].copy()

if not orders_entered_repayment.empty:
    timeframes = [5, 7, 10, 15]
    results = {}

    # --- Define user segments and corresponding dataframes --- #
    segments = {
        'All Users': orders_entered_repayment,
        'New Users': orders_entered_repayment[orders_entered_repayment['user_type'] == 'New'],
        'Existing Users': orders_entered_repayment[orders_entered_repayment['user_type'] == 'Existing']
    }

    # --- Calculate proportions for each segment --- #
    for segment_name, segment_df in segments.items():
        total_in_segment = len(segment_df)
        if total_in_segment > 0:
            proportions = {}
            print(f"\n-- {segment_name} (Total: {total_in_segment}) --")
            for days in timeframes:
                count = len(segment_df[segment_df['time_to_repayment_start'] <= days])
                prop = count / total_in_segment
                proportions[f'<= {days} Days'] = prop
                print(f"  Proportion within {days} days: {prop:.1%}")
            results[segment_name] = proportions
        else:
            print(f"\n-- {segment_name}: No orders found entering repayment. --")
            results[segment_name] = {f'<= {d} Days': 0 for d in timeframes} # Add zeros if no data

    # --- Combine results into a DataFrame and format --- #
    summary_df = pd.DataFrame(results)
    summary_df.index.name = 'Timeframe'

    print("\n--- Combined Repayment Timing Summary --- ")
    print(summary_df.to_string(float_format='{:.1%}'.format)) # Print formatted table to console

    # Format for PNG saving (map applies formatting)
    summary_df_formatted = summary_df.applymap('{:.1%}'.format)
    save_df_as_png(summary_df_formatted, os.path.join(pipeline_dir, 'repayment_timing_cumulative_proportion_segmented.png'), title='Cumulative Proportion of Orders Entering Repayment by User Type')

else:
    print("No orders found entering repayment to analyze timing.")
# --------------------------------

# --- User Spending Analysis (Segmented) ---
print("\n=== User Spending Analysis (Segmented) ===")
spending_dir = output_dirs['spending']

# --- Requested Amount Analysis ---
print("\n--- Requested Amount Analysis ---")
# Calculate total requested amount per user
user_requested_amount = orders.groupby('user_id')['amount'].sum().reset_index()

# Merge user_type (Need the first user_type associated with each user_id)
user_type_map = orders[['user_id', 'user_type']].drop_duplicates('user_id').set_index('user_id')
user_requested_amount = user_requested_amount.join(user_type_map, on='user_id')

print("\nUser Requested Amount Statistics by User Type:")
user_requested_stats = user_requested_amount.groupby('user_type')['amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(user_requested_stats)
save_df_as_png(user_requested_stats.round(2), os.path.join(spending_dir, 'user_requested_amount_stats_by_user_type.png'), title='User Requested Amount Statistics By User Type')


# Create detailed distribution plot (Segmented Requested Amount)
fig_req, axes_req = plt.subplots(1, 2, figsize=(15, 5))

# Define categories, palette, colormap, and medians needed for plot order and lines
categories_req = sorted(user_requested_amount['user_type'].dropna().unique())
palette_req = sns.color_palette("husl", len(categories_req))
color_map_req = dict(zip(categories_req, palette_req))
medians_req = user_requested_amount.groupby('user_type')['amount'].median()

sns.histplot(data=user_requested_amount, x='amount', hue='user_type', bins=50, kde=True, element="step", ax=axes_req[0], hue_order=categories_req)
axes_req[0].set_title('Distribution Of Total Requested Amount Per User')
axes_req[0].set_xlabel('Total Requested Amount ($)')
axes_req[0].grid(True, axis='x')

sns.violinplot(data=user_requested_amount, y='amount', x='user_type', ax=axes_req[1], order=categories_req) # Explicit order
axes_req[1].set_title('Violin Plot Of Total Requested Amount Per User')
axes_req[1].set_ylabel('Total Requested Amount ($)')
axes_req[1].set_xlabel('User Type') # Add x-axis label
axes_req[1].grid(True, axis='y')

# Add median lines for Requested Amount Violin Plot
print("\n--- Plotting Medians for Requested Amount ---") # DEBUG
for cat in categories_req:
    if cat in medians_req.index:
        median_val = medians_req[cat]
        line_color = color_map_req[cat]
        # --- DEBUG PRINT --- 
        print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
        # -------------------
        axes_req[1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(spending_dir, 'user_requested_amount_detailed_segmented.png'))
plt.close(fig_req) # Close plot

# --- Approved Amount Analysis ---
print("\n--- Approved Amount Analysis ---")
# Filter for approved orders
approved_orders = orders[orders['approval_type'].str.startswith('Approved', na=False)].copy()

if not approved_orders.empty:
    # Calculate total approved amount per user
    user_approved_amount = approved_orders.groupby('user_id')['amount'].sum().reset_index()
    user_approved_amount = user_approved_amount.join(user_type_map, on='user_id') # Merge user_type

    print("\nUser Approved Amount Statistics by User Type:")
    user_approved_stats = user_approved_amount.groupby('user_type')['amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(user_approved_stats)
    save_df_as_png(user_approved_stats.round(2), os.path.join(spending_dir, 'user_approved_amount_stats_by_user_type.png'), title='User Approved Amount Statistics By User Type')

    # Create detailed distribution plot (Segmented Approved Amount)
    fig_appr, axes_appr = plt.subplots(1, 2, figsize=(15, 5))

    # Define categories, palette, colormap, and medians needed for plot order and lines
    categories_appr = sorted(user_approved_amount['user_type'].dropna().unique())
    palette_appr = sns.color_palette("husl", len(categories_appr))
    color_map_appr = dict(zip(categories_appr, palette_appr))
    medians_appr = user_approved_amount.groupby('user_type')['amount'].median()

    sns.histplot(data=user_approved_amount, x='amount', hue='user_type', bins=50, kde=True, element="step", ax=axes_appr[0], hue_order=categories_appr)
    axes_appr[0].set_title('Distribution Of Total Approved Amount Per User')
    axes_appr[0].set_xlabel('Total Approved Amount ($)')
    axes_appr[0].grid(True, axis='x')

    sns.violinplot(data=user_approved_amount, y='amount', x='user_type', ax=axes_appr[1], order=categories_appr) # Explicit order
    axes_appr[1].set_title('Violin Plot Of Total Approved Amount Per User')
    axes_appr[1].set_ylabel('Total Approved Amount ($)')
    axes_appr[1].set_xlabel('User Type')
    axes_appr[1].grid(True, axis='y')

    # Add median lines for Approved Amount Violin Plot
    print("\n--- Plotting Medians for Approved Amount ---") # DEBUG
    for cat in categories_appr:
        if cat in medians_appr.index:
            median_val = medians_appr[cat]
            line_color = color_map_appr[cat]
            # --- DEBUG PRINT --- 
            print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
            # -------------------
            axes_appr[1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(spending_dir, 'user_approved_amount_detailed_segmented.png'))
    plt.close(fig_appr)
else:
    print("No approved orders found to analyze.")

# --- Repayment Amount Analysis ---
print("\n--- Amount Entering Repayment Analysis ---")
# Filter for orders entering repayment
repayment_orders = orders[orders['repayment_ts'].notna()].copy()

if not repayment_orders.empty:
    # Calculate total amount entering repayment per user
    user_repayment_amount = repayment_orders.groupby('user_id')['amount'].sum().reset_index()
    user_repayment_amount = user_repayment_amount.join(user_type_map, on='user_id') # Merge user_type

    print("\nUser Amount Entering Repayment Statistics by User Type:")
    user_repayment_stats = user_repayment_amount.groupby('user_type')['amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(user_repayment_stats)
    save_df_as_png(user_repayment_stats.round(2), os.path.join(spending_dir, 'user_repayment_amount_stats_by_user_type.png'), title='User Amount Entering Repayment Statistics By User Type')

    # Create detailed distribution plot (Segmented Repayment Amount)
    fig_rep, axes_rep = plt.subplots(1, 2, figsize=(15, 5))

    # Define categories, palette, colormap, and medians needed for plot order and lines
    categories_rep = sorted(user_repayment_amount['user_type'].dropna().unique())
    palette_rep = sns.color_palette("husl", len(categories_rep))
    color_map_rep = dict(zip(categories_rep, palette_rep))
    medians_rep = user_repayment_amount.groupby('user_type')['amount'].median()

    sns.histplot(data=user_repayment_amount, x='amount', hue='user_type', bins=50, kde=True, element="step", ax=axes_rep[0], hue_order=categories_rep) # Explicit hue order
    axes_rep[0].set_title('Distribution Of Total Amount Entering Repayment Per User')
    axes_rep[0].set_xlabel('Total Amount Entering Repayment ($)')
    axes_rep[0].grid(True, axis='x')

    sns.violinplot(data=user_repayment_amount, y='amount', x='user_type', ax=axes_rep[1], order=categories_rep) # Explicit order
    axes_rep[1].set_title('Violin Plot Of Total Amount Entering Repayment Per User')
    axes_rep[1].set_ylabel('Total Amount Entering Repayment ($)')
    axes_rep[1].set_xlabel('User Type')
    axes_rep[1].grid(True, axis='y')

    # Add median lines for Repayment Amount Violin Plot
    print("\n--- Plotting Medians for Repayment Amount ---") # DEBUG
    for cat in categories_rep:
        if cat in medians_rep.index:
            median_val = medians_rep[cat]
            line_color = color_map_rep[cat]
            # --- DEBUG PRINT --- 
            print(f"  Category='{cat}', Median={median_val:.2f}, Color={line_color}")
            # -------------------
            axes_rep[1].axhline(median_val, color=line_color, linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(spending_dir, 'user_repayment_amount_detailed_segmented.png'))
    plt.close(fig_rep)
else:
    print("No orders entering repayment found to analyze.")

print(f"\nAnalysis complete. All plots and tables saved to subdirectories within '{output_base}'.")

# # Calculate total amount spent per user
# print("\n=== User Spending Analysis ===")
# user_spending = orders.groupby('user_id')['amount'].sum().reset_index()
# print("\nUser Spending Statistics:")
# print(user_spending['amount'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# # Create violin plot of user spending
# plt.figure(figsize=(15, 5))
# sns.violinplot(y=user_spending['amount'])
# plt.title('Distribution of Total Amount Spent per User')
# plt.ylabel('Total Amount ($)')
# plt.grid(True, axis='y')

# plt.tight_layout()
# plt.savefig('output/user_spending_distribution.png')

# # Create detailed distribution plot
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# sns.histplot(user_spending['amount'], bins=50)
# plt.title('Distribution of Total Amount Spent per User')
# plt.xlabel('Total Amount ($)')
# plt.grid(True, axis='x')

# plt.subplot(1, 2, 2)
# sns.violinplot(y=user_spending['amount'])
# plt.title('Violin Plot of Total Amount Spent per User')
# plt.ylabel('Total Amount ($)')
# plt.grid(True, axis='y')

# plt.tight_layout()
# plt.savefig('output/user_spending_detailed.png')






