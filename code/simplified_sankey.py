import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

# Function to lighten/desaturate hex color (simple approach) - *No longer needed for this version, but kept for potential reuse*
def modify_color(hex_color, factor=0.7):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        print(f"Warning: Invalid hex color '{hex_color}' received in modify_color")
        return '#CCCCCC'
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = tuple(min(255, int(c + (255 - c) * (1 - factor))) for c in rgb)
        return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'
    except ValueError:
        print(f"Warning: ValueError processing hex color '{hex_color}' in modify_color")
        return '#CCCCCC'

# Helper function to convert hex to rgba for Plotly
def hex_to_rgba(hex_color, alpha=0.8):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        print(f"Warning: Invalid hex color '{hex_color}' received in hex_to_rgba")
        return f'rgba(204, 204, 204, {alpha})' # Default grey
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    except ValueError:
        print(f"Warning: ValueError processing hex color '{hex_color}' in hex_to_rgba")
        return f'rgba(204, 204, 204, {alpha})' # Default grey

# Load the data
order_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/order_dataset.csv'
orders = pd.read_csv(order_file)
# user_file = 'Perpay_Strategic_Analytics_Data_Challenge_B_2025/user_dataset.csv' # User data not needed for total funnel
# users = pd.read_csv(user_file)

# Get set of existing user IDs - *No longer needed*
# existing_user_ids = set(users['user_id'].unique())

# Categorize orders - *No longer needed*
# orders['is_existing_user'] = orders['user_id'].isin(existing_user_ids)

# Convert timestamp columns to datetime
timestamp_cols = ['application_start_ts', 'application_complete_ts', 'awaiting_payment_ts', 'repayment_ts']
for col in timestamp_cols:
    orders[col] = pd.to_datetime(orders[col])

# Calculate total applications started (needed for percentages)
total_applications = len(orders[orders['application_start_ts'].notna()])

# --- Redefine Nodes, Labels, Colors, Positions for Single Funnel ---

labels = [
    "Application Start", "Completed Application", "Approved",
    "Unfinished", "Not Approved", "No Deposit", "Deposit Setup"
]

colors = {
    "Application Start": '#2E86C1',
    "Completed Application": '#3498DB',
    "Approved": '#2980B9',
    "Unfinished": '#E74C3C',  # Changed to red for attrition
    "Not Approved": '#E74C3C',
    "No Deposit": '#F39C12',
    "Deposit Setup": '#1ABC9C'
}
node_colors = [colors[lbl] for lbl in labels]

# Define X positions (main flow left/mid, attrition/end right)
x_map = {
    "Application Start": 0.01,
    "Completed Application": 0.35,
    "Approved": 0.65,
    "Unfinished": 0.99,
    "Not Approved": 0.99,
    "No Deposit": 0.99,
    "Deposit Setup": 0.99
}
x_positions = [x_map[lbl] for lbl in labels]

# Define Y positions (increased separation for Not Approved flow)
y_map = {
    "Application Start": 0.5,       # Lowered main flow
    "Completed Application": 0.5,   # Lowered main flow
    "Approved": 0.5,              # Lowered main flow
    "Unfinished": 0.05,           # Top attrition from Start (Keep high)
    "Not Approved": 0.3,           # Attrition from Comp App (Positioned clearly above main flow)
    "No Deposit": 0.65,           # Attrition from Approved (Positioned below main flow)
    "Deposit Setup": 0.95         # Success from Approved (Positioned further below)
}
y_positions = [y_map[lbl] for lbl in labels]


# Create a mapping of labels to indices
label_indices = {label: idx for idx, label in enumerate(labels)}

# --- Recalculate Flows for Single Funnel ---

source = []
target = []
value = []
link_colors = []
node_counts_actual = {} # Store actual counts for each node

# DataFrames for each stage (total counts)
start_total = orders[orders['application_start_ts'].notna()]
complete_total = orders[orders['application_complete_ts'].notna()]
approved_total = orders[orders['awaiting_payment_ts'].notna()]
deposit_total = orders[orders['repayment_ts'].notna()]

# Populate initial node counts
node_counts_actual[label_indices["Application Start"]] = len(start_total)
node_counts_actual[label_indices["Completed Application"]] = len(complete_total)
node_counts_actual[label_indices["Approved"]] = len(approved_total)
node_counts_actual[label_indices["Deposit Setup"]] = len(deposit_total)


# Helper function to add links
def add_link(src_label, tgt_label, count):
    if count > 0:
        src_idx = label_indices[src_label]
        tgt_idx = label_indices[tgt_label]
        source.append(src_idx)
        target.append(tgt_idx)
        value.append(count)
        # Use target color for attrition flows for better visual distinction
        alpha_value = 0.8 # Define transparency level
        if tgt_label in ["Unfinished", "Not Approved", "No Deposit"]:
             link_colors.append(hex_to_rgba(colors[tgt_label], alpha=alpha_value))
        else:
             link_colors.append(hex_to_rgba(colors[src_label], alpha=alpha_value))


# Flow 1: Start -> Complete / Unfinished
complete_count = len(start_total[start_total['order_id'].isin(complete_total['order_id'])])
unfinished_count = len(start_total) - complete_count
add_link("Application Start", "Completed Application", complete_count)
add_link("Application Start", "Unfinished", unfinished_count)
node_counts_actual[label_indices["Unfinished"]] = unfinished_count


# Flow 2: Complete -> Approved / Not Approved
approved_count = len(complete_total[complete_total['order_id'].isin(approved_total['order_id'])])
not_approved_count = len(complete_total) - approved_count
add_link("Completed Application", "Approved", approved_count)
add_link("Completed Application", "Not Approved", not_approved_count)
node_counts_actual[label_indices["Not Approved"]] = not_approved_count


# Flow 3: Approved -> Deposit Setup / No Deposit
deposit_count = len(approved_total[approved_total['order_id'].isin(deposit_total['order_id'])])
no_deposit_count = len(approved_total) - deposit_count
add_link("Approved", "Deposit Setup", deposit_count)
add_link("Approved", "No Deposit", no_deposit_count)
node_counts_actual[label_indices["No Deposit"]] = no_deposit_count


# Update labels with specific node counts and percentages of GRAND TOTAL
updated_labels = []
for i, label in enumerate(labels):
    count = node_counts_actual.get(i, 0) # Get count for this specific node index
    percentage = (count / total_applications) * 100 if total_applications > 0 else 0
    # Shorten base label for display if needed
    display_label = label.replace("Application ", "App ") \
                         .replace("Completed Application", "Comp App")
    updated_labels.append(f"{display_label}<br>({count:,} | {percentage:.1f}%)")

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    arrangement='freeform', # Changed arrangement back to freeform
    node=dict(
        pad=20, # Adjusted padding
        thickness=20, # Adjusted thickness
        line=dict(color="black", width=0.5),
        label=updated_labels,
        color=node_colors,
        x=x_positions,
        y=y_positions
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors # Use calculated link colors with transparency
    )
)])

fig.update_layout(
    title_text="Total Application Funnel<br>Counts and Percentages of Total Applicants", # Updated title
    font_size=10, # Adjusted font size
    height=700, # Adjusted height
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Define output base directory and subdirectories
output_base = 'output/sankey'
html_dir = os.path.join(output_base, 'html')
png_dir = os.path.join(output_base, 'png')

# Create directories if they don't exist
os.makedirs(html_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# Define output filenames
base_filename = 'total_funnel_sankey' # Changed base filename
html_path = os.path.join(html_dir, f"{base_filename}.html")
png_path = os.path.join(png_dir, f"{base_filename}.png")


# Save the figure
pio.write_html(fig, html_path)
pio.write_image(fig, png_path, scale=2) 