import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

###########################################
# ### 3.Extraction and transformation of the “Maximum individual wave height” time series. Data extraction.
###########################################

final_csv_file_path = 'final_dataset.csv'
range_limit_of_selected_data = '2024-12-31'  # ONLY TAKE DATA UNTIL 2024
max_wave_height_var = 'Maximum individual wave height'

df_data = pd.read_csv(final_csv_file_path, index_col=0)
df_data = df_data.sort_index()


###########################################
# #### **4.2. Correlation analysis of the variable ‘Maximum individual wave height’ with other climatic variables.**
###########################################

# Compute correlation using Pearson, Spearman, and Kendall methods
correlation_pearson = df_data.corr(method='pearson')[max_wave_height_var]
correlation_spearman = df_data.corr(method='spearman')[max_wave_height_var]
correlation_kendall = df_data.corr(method='kendall')[max_wave_height_var]

# Combine results into a single DataFrame
correlation_results = pd.DataFrame({
    'Pearson': correlation_pearson,
    'Spearman': correlation_spearman,
    'Kendall': correlation_kendall
}).drop(index=max_wave_height_var)  # Drop the target variable itself

# Drop any rows (variables) with NaN values in the entire dataset.
correlation_results = correlation_results.dropna(axis=0, how='any')


# Define correlation strength categories
def categorize_correlation(value):
    if abs(value) >= 0.7:
        return "Strong"
    elif abs(value) >= 0.4:
        return "Moderate"
    elif abs(value) >= 0.2:
        return "Weak"
    else:
        return "None"

# Apply categorization
correlation_results['Pearson Category'] = correlation_results['Pearson'].apply(categorize_correlation)
correlation_results['Spearman Category'] = correlation_results['Spearman'].apply(categorize_correlation)
correlation_results['Kendall Category'] = correlation_results['Kendall'].apply(categorize_correlation)

# Sort correlation results by Pearson correlation in descending order
correlation_results_sorted = correlation_results.sort_values(by='Pearson', ascending=False)

# Define the path for saving the correlation results
correlation_results_csv_path = "02 - correlation_results_sorted.csv"

# Save the sorted correlation results to a CSV file
correlation_results_sorted.to_csv(correlation_results_csv_path)

# Reorder variables based on Pearson correlation strength
ordered_vars = correlation_results_sorted.index.insert(0, max_wave_height_var)  # Insert target variable at the top

# Function to plot correlation bars with proper coloring
def plot_correlation_bars(correlation_data, title, filename):
    # Increase the figure size for more space
    plt.figure(figsize=(18, 25))  # Larger figure width and height for better clarity
    
    # Use a red-to-blue colormap for the bars
    norm = Normalize(vmin=-1, vmax=1)  # Set the range of correlation values
    cmap = cm.get_cmap('coolwarm')  # Red to blue colormap
    
    # Plot bars for correlations with significantly larger bar height
    #bar = plt.barh(correlation_data.index, correlation_data, color=cmap(norm(correlation_data)), height=0.9)
    bar = plt.barh(correlation_data.index, correlation_data, color=cmap(norm(correlation_data)), height=0.9)
    
    # Annotate with the correlation values (decreasing fontsize for better readability)
    for rect in bar:
        plt.text(
            rect.get_width() + 0.01, rect.get_y() + rect.get_height() / 2,
            f'{rect.get_width():.2f}', color='black', va='center', fontsize=14  # Adjusted fontsize for correlation labels
        )
        y_center = rect.get_y() + rect.get_height() / 2
        plt.plot([-0.02, 0], [y_center, y_center], color='gray', linewidth=1, linestyle='--')  # Línea horizontal auxiliar

    # Set plot titles and labels with increased spacing
    plt.title(f'{title} Correlations with {max_wave_height_var}', fontsize=22)
    plt.xlabel('Correlation Coefficient', fontsize=18)
    plt.ylabel('ERA5 dataset variables', fontsize=18)
    plt.tick_params(axis='y', pad=10)
    
    # Adjust layout to prevent overlapping and ensure more space between labels
    plt.tight_layout()

    # Increase the padding on the Y-axis to prevent label overlap
    plt.subplots_adjust(left=0.41, right=0.95, top=0.98, bottom=0.04)  # Further left margin to prevent cut-off

    # Rotate Y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=16)  # Adjust this for more spacing
    plt.xticks(fontsize=18)

    # Save and display the plot
    plt.savefig(filename, dpi=300)
    plt.show()

# Pearson correlation bars
plot_correlation_bars(
    correlation_results_sorted['Pearson'], 
    'Pearson', 
    'pearson_correlation.png'
)


# Spearman correlation bars
plot_correlation_bars(
    correlation_results_sorted['Spearman'], 
    'Spearman', 
    'spearman_correlation.png'
)

# Kendall correlation bars
plot_correlation_bars(
    correlation_results_sorted['Kendall'], 
    'Kendall', 
    'kendall_correlation.png'
)
