import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines  # To add the mean line to the legend
from datetime import datetime as dt
import numpy as np
import contextily as ctx
import os



####### LOADING DATA ###########

def process_and_reverse_columns(file_path, dates, building_tree_stats=['mean', 'min', 'max']):
    """
    Load the dataset, clean the GeoDataFrame, reverse the values in specific columns, and return 
    the cleaned GeoDataFrame and a melted DataFrame.
    
    Parameters:
    file_path (str): Path to the dataset (GeoPackage file).
    dates (list): A list of dates in 'YYYYMMDD' format.
    building_tree_stats (list): The statistics to reverse for both 'building' and 'tree'.

    Returns:
    gdf_cleaned (GeoDataFrame): The cleaned GeoDataFrame with reversed columns.
    df (DataFrame): The melted DataFrame with separated 'Metric', 'Category', and 'Date' columns.
    """
    
    # Load the dataset
    gdf = gpd.read_file(file_path, engine='pyogrio')

    # Check for null values and clean the DataFrame
    gdf_cleaned = gdf.dropna(subset=[f'{dates[0]}_building_mean'])  # Adjust column as necessary

    # Reverse columns based on the dates and statistics
    columns_to_reverse = []
    for date in dates:
        for stat in building_tree_stats:
            columns_to_reverse.append(f'{date}_building_{stat}')
            columns_to_reverse.append(f'{date}_tree_{stat}')

    # Reverse the values in each column
    for column in columns_to_reverse:
        if column in gdf_cleaned.columns:  # Ensure the column exists in the DataFrame
            gdf_cleaned[column] = 1 - gdf_cleaned[column]

    # Create a list of columns to melt for boxplot
    columns_to_melt = []
    for date in dates:
        columns_to_melt.extend([
            f'{date}_building_mean', f'{date}_building_std',
            f'{date}_tree_mean', f'{date}_tree_std'
        ])

    # Reshape the GeoDataFrame to long format
    df = pd.melt(gdf_cleaned[columns_to_melt], var_name='Metric_Category', value_name='Value')

    # Split 'Metric_Category' into separate 'Metric', 'Category', and 'Date' columns
    df[['Date', 'Category', 'Metric']] = df['Metric_Category'].str.split('_', expand=True)

    # Drop the original 'Metric_Category' column
    df.drop(columns=['Metric_Category'], inplace=True)

    # Dictionary of replacements
    replacements = {
        'building': 'Buildings only',
        'tree': 'Buildings and trees',
    }

    # Replace values in 'Category'
    df['Category'] = df['Category'].replace(replacements)

    return gdf_cleaned, df


def melt_gdf_for_plotting(gdf, dates, metric_prefixes=['building', 'tree'], metrics=['mean', 'std']):
    """
    Converts a GeoDataFrame into a melted DataFrame suitable for plotting. The function filters the
    specified columns based on the provided dates and metric types, then reshapes the DataFrame into a
    long format and splits columns to extract the date, category, and metric information.

    Parameters:
    gdf (GeoDataFrame): The original GeoDataFrame to reshape.
    dates (list of str): A list of date strings to filter columns by, e.g., ['20240620', '20240621'].
    metric_prefixes (list of str): Prefixes for the categories, default is ['building', 'tree'].
    metrics (list of str): Metrics to include, default is ['mean', 'std'].

    Returns:
    pd.DataFrame: A melted DataFrame with separate 'Date', 'Category', and 'Metric' columns.
    """

    # Create a list of columns to melt for boxplot based on dates, prefixes, and metrics
    columns_to_melt = []
    for date in dates:
        for prefix in metric_prefixes:
            for metric in metrics:
                columns_to_melt.append(f'{date}_{prefix}_{metric}')
    
    # Ensure that the columns exist in the gdf before attempting to melt
    available_columns = [col for col in columns_to_melt if col in gdf.columns]

    if not available_columns:
        raise ValueError("None of the specified columns exist in the GeoDataFrame.")
    
    # Reshape the GeoDataFrame to long format
    df_melted = pd.melt(gdf[available_columns], var_name='Metric_Category', value_name='Value')

    # Split 'Metric_Category' into separate 'Date', 'Category', and 'Metric' columns
    df_melted[['Date', 'Category', 'Metric']] = df_melted['Metric_Category'].str.split('_', expand=True)

    # Drop the original 'Metric_Category' column
    df_melted.drop(columns=['Metric_Category'], inplace=True)

    # Dictionary of replacements for more readable categories
    category_replacements = {
        'building': 'Buildings only',
        'tree': 'Buildings and trees'
    }

    # Replace values in 'Category'
    df_melted['Category'] = df_melted['Category'].replace(category_replacements)

    return df_melted

def preprocess_amsterdam_data(gdf_cleaned, buurten_path, values_to_keep=None):
    """
    Preprocess the GeoDataFrame for the Amsterdam dataset by filtering specific values in the 'Gebruiksfunctie' column,
    replacing Dutch categories with English equivalents, reprojecting to match the CRS of the neighborhoods data,
    and performing a spatial join with neighborhood attributes.
    
    Parameters:
    gdf_cleaned (GeoDataFrame): The cleaned GeoDataFrame to be preprocessed.
    buurten_path (str): The path to the neighborhoods GeoJSON file.
    values_to_keep (list): A list of values in 'Gebruiksfunctie' column to filter by. If None, defaults to specific values for Amsterdam.
    
    Returns:
    filtered_gdf (GeoDataFrame): The preprocessed GeoDataFrame with neighborhood attributes added.
    """
    if values_to_keep is None:
        values_to_keep = ['Voetpad', 'Voetpad op trap', 'Voetgangersgebied', 'Fietspad', 'Rijbaan']
    
    # Filter the GeoDataFrame based on 'Gebruiksfunctie' values
    filtered_gdf = gdf_cleaned[gdf_cleaned['Gebruiksfunctie'].isin(values_to_keep)]
    
    # Dictionary of replacements for Dutch to English categories
    replacements = {
        'Voetpad': 'Sidewalk',
        'Voetpad op trap': 'Sidewalk',
        'Voetgangersgebied': 'Sidewalk',
        'Fietspad': 'Cycle lane',
        'Rijbaan': 'Road',
    }
    
    # Replace the values in 'Gebruiksfunctie' column
    filtered_gdf['Gebruiksfunctie'] = filtered_gdf['Gebruiksfunctie'].replace(replacements)
    
    # Load the neighborhoods data
    buurten = gpd.read_file(buurten_path)
    
    # Reproject filtered_gdf to match buurten CRS
    filtered_gdf = filtered_gdf.to_crs(buurten.crs)
    
    # Perform a spatial join with neighborhoods data
    filtered_gdf = gpd.sjoin(
        filtered_gdf, buurten[['geometry', 'CBS_Buurtcode', 'Buurtcode', 'Buurt', 'Wijkcode', 'Wijk', 
                               'Gebiedcode', 'Gebied', 'Stadsdeelcode', 'Stadsdeel']],
        how='left', predicate='intersects'
    )
    
    # Reset index after join
    filtered_gdf.reset_index(inplace=True)
    
    return filtered_gdf



def calculate_daily_average_shade(filtered_gdf, time_increments, date, avg_column_name):
    """
    Calculate the daily average of tree shade percentage across specific time increments for a given date.
    
    Parameters:
    filtered_gdf (GeoDataFrame): The GeoDataFrame containing shade percentage data.
    time_increments (list): A list of time increments as strings (e.g., ['1130', '1200', '1230']).
    date (str): The date in 'YYYYMMDD' format (e.g., '20240620').
    avg_column_name (str): The name for the new column to store the daily average.
    
    Returns:
    filtered_gdf (GeoDataFrame): The GeoDataFrame with the new column containing the daily average shade percentage.
    """
    # Create a list of columns based on the date and time increments
    columns_to_average = [f'{date}_tree_shade_percent_at_{time}' for time in time_increments]
    
    # Calculate the average across the specified columns
    filtered_gdf[avg_column_name] = filtered_gdf[columns_to_average].mean(axis=1)
    
    return filtered_gdf


def calculate_shade_score(filtered_gdf, time_increments, date, score_column_name, threshold=30):
    """
    Calculate the score of time increments meeting or exceeding a shade percentage threshold 
    for a given date and add it as a new column in the GeoDataFrame.

    Parameters:
    filtered_gdf (GeoDataFrame): The GeoDataFrame containing shade percentage data.
    time_increments (list): A list of time increments as strings (e.g., ['1130', '1200', '1230']).
    date (str): The date in 'YYYYMMDD' format (e.g., '20240620').
    score_column_name (str): The name for the new column to store the shade score.
    threshold (float): The shade percentage threshold (e.g., 30 for 30%).

    Returns:
    filtered_gdf (GeoDataFrame): The GeoDataFrame with the new column containing the shade score.
    """
    # Create a list of columns based on the date and time increments
    columns_to_check = [f'{date}_tree_shade_percent_at_{time}' for time in time_increments]

    # Calculate the score as the fraction of times the shade percentage is above the threshold
    filtered_gdf[score_column_name] = (filtered_gdf[columns_to_check] >= threshold).sum(axis=1) / len(time_increments)

    return filtered_gdf

########### PLOTTING #######

def plot_shade_metrics(df, background='dark', title='Daily Shade Percentage on All Surfaces', save_path=None, file_name=None):
    """
    Plots boxplots of shade metrics for buildings and trees across different dates, focusing on the 'mean' metric.
    Optionally saves the plot to a specified folder with a chosen name.
    
    Parameters:
    df (DataFrame): The reshaped DataFrame containing 'Metric', 'Value', 'Category', and 'Date'.
    background (str): Either 'dark' or 'light' to set the plot background.
    save_path (str): Path to the folder where the plot should be saved. If None, the plot is not saved.
    file_name (str): Name for the saved plot file. If None, the plot is not saved.
    """
    # Filter the DataFrame to only include 'mean' metrics
    df_mean = df[df['Metric'] == 'mean']
    # Set the plot style based on the background parameter
    if background == 'dark':
        sns.set_style("darkgrid")
        text_color = 'white'
        palette = {'Buildings only': '#7f8c8d', 'Buildings and trees': '#2ecc71'}  # Grey and Green
        fig_color = '#2e2e2e'  # Dark background for figure
    else:
        sns.set_style("whitegrid")
        text_color = 'black'
        palette = {'Buildings only': '#7f8c8d', 'Buildings and trees': '#73d016'}  # Darker grey and Green
        fig_color = 'white'  # Light background for figure

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set figure background color
    fig.patch.set_facecolor(fig_color)
    ax.set_facecolor(fig_color if background == 'light' else '#1a1a1a')  # Darker plot area for dark background

    # Create the boxplot with the custom palette
    # sns.boxplot(x='Date', y='Value', hue='Category', data=df_mean, palette=palette, ax=ax)

    sns.violinplot(x='Date', y='Value', hue='Category', data=df_mean, palette=palette, ax=ax, split=True, gap=.1, inner="quart")

    # Annotate each boxplot with mean and median values
    for i, (date, category) in enumerate([(d, c) for d in df_mean['Date'].unique() for c in df_mean['Category'].unique()]):
        subset = df_mean[(df_mean['Date'] == date) & (df_mean['Category'] == category)]['Value']
        if subset.empty:
            continue

        # Calculate statistics
        median_value = np.median(subset)
        mean_value = np.mean(subset)

        # Get x position of the current boxplot
        x_position = i // len(df_mean['Category'].unique()) + (i % len(df_mean['Category'].unique())) * 0.4 - 0.18

        # Annotate with the calculated values
        ax.text(x_position, median_value, f'Median: {median_value*100:.0f}%', ha='center', va='bottom', color=text_color, fontsize=10, weight='bold')
        # ax.text(x_position, mean_value, f'Mean: {mean_value*100:.0f}%', ha='center', va='top', color='yellow' if background == 'dark' else 'black', fontsize=10)
    
    # Customize the plot
    ax.set_title(title, color=text_color, fontsize=18)
    ax.set_xlabel('Date', color=text_color)
    ax.set_ylabel('Daily Shade', color=text_color)

    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Customize the axes
    ax.spines['top'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

    # Adjust the legend
    legend = ax.legend(title='', frameon=False, loc='lower center', ncol=2)
    plt.setp(legend.get_title(), color=text_color)
    for text in legend.get_texts():
        text.set_color(text_color)

    # Save the plot if the path and file name are provided
    if save_path and file_name:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(full_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved at {full_path}")
    
    # Show the plot
    plt.show()


def plot_shade_metrics_for_cities(dfs, cities, background='dark', title='Daily Shade Percentage on All Surfaces (20240620)', save_path=None, file_name=None):
    """
    Plots violin plots of shade metrics for multiple cities for a single day, focusing on the 'mean' metric.
    Optionally saves the plot to a specified folder with a chosen name.

    Parameters:
    dfs (list of DataFrame): List of DataFrames for each city, containing 'Metric', 'Value', 'Category', and 'Date'.
    cities (list of str): List of city names corresponding to each DataFrame.
    background (str): Either 'dark' or 'light' to set the plot background.
    save_path (str): Path to the folder where the plot should be saved. If None, the plot is not saved.
    file_name (str): Name for the saved plot file. If None, the plot is not saved.
    """
    # Combine DataFrames and add a 'City' column
    for i, df in enumerate(dfs):
        df['City'] = cities[i]
    
    # Concatenate the DataFrames into one
    combined_df = pd.concat(dfs)
    
    # Filter the DataFrame to include only 'mean' metrics and the specific date (20240620)
    df_filtered = combined_df[(combined_df['Metric'] == 'mean') & (combined_df['Date'] == '20240620')]

    # Set the plot style based on the background parameter
    if background == 'dark':
        sns.set_style("darkgrid")
        text_color = 'white'
        palette = {'Buildings only': '#7f8c8d', 'Buildings and trees': '#2ecc71'}  # Grey and Green
        fig_color = '#2e2e2e'  # Dark background for figure
    else:
        sns.set_style("whitegrid")
        text_color = 'black'
        palette = {'Buildings only': '#7f8c8d', 'Buildings and trees': '#73d016'}  # Darker grey and Green
        fig_color = 'white'  # Light background for figure

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set figure background color
    fig.patch.set_facecolor(fig_color)
    ax.set_facecolor(fig_color if background == 'light' else '#1a1a1a')  # Darker plot area for dark background

    # Create the violin plot with the custom palette
    sns.violinplot(x='City', y='Value', hue='Category', data=df_filtered, palette=palette, ax=ax, split=True, gap=.1, inner="quart")

    # Annotate each violin plot with mean and median values
    for i, (city, category) in enumerate([(city, c) for city in df_filtered['City'].unique() for c in df_filtered['Category'].unique()]):
        subset = df_filtered[(df_filtered['City'] == city) & (df_filtered['Category'] == category)]['Value']
        if subset.empty:
            continue

        # Calculate statistics
        median_value = np.median(subset)
        mean_value = np.mean(subset)

        # Get x position of the current violin plot
        x_position = i // len(df_filtered['Category'].unique()) + (i % len(df_filtered['Category'].unique())) * 0.4 - 0.18

        # Annotate with the calculated values
        ax.text(x_position, median_value, f'Median: {median_value*100:.0f}%', ha='center', va='bottom', color=text_color, fontsize=10, weight='bold')
    
    # Customize the plot
    ax.set_title(title, color=text_color, fontsize=18)
    ax.set_xlabel('City', color=text_color)
    ax.set_ylabel('Daily Shade', color=text_color)

    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Customize the axes
    ax.spines['top'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)

    # Adjust the legend
    legend = ax.legend(title='', frameon=False, loc='lower center', ncol=2)
    plt.setp(legend.get_title(), color=text_color)
    for text in legend.get_texts():
        text.set_color(text_color)

    # Save the plot if the path and file name are provided
    if save_path and file_name:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(full_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved at {full_path}")
    
    # Show the plot
    plt.show()


def plot_shade_comparison(datasets, dataset_names, dates, times, osmid, title, save_path=None, meanline_color='orange'):
    """
    Plot a comparison of shade percentages across multiple datasets for specific dates and times.
    
    Parameters:
    datasets (list): A list of datasets (DataFrames or GeoDataFrames) to compare.
    dataset_names (list): Names of each dataset (for plot titles).
    dates (list): List of date strings in 'YYYYMMDD' format.
    times (list): List of time increments (e.g., ['1130', '1200', '1230']).
    osmid (str): The OSM ID for labeling the plot.
    title (str): Title for the plot.
    save_path (str): Path to save the plot (optional).
    meanline_color (str): Color for the mean line in the plot (default: 'orange').
    """
    # Prepare the subplot grid (one subplot per dataset)
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 4), sharey=True)  # Sharing y-axis for comparison

    if num_datasets == 1:
        axes = [axes]  # Convert single axis to list for uniform handling

    # Loop through each dataset and its corresponding name
    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
        plot_data = []

        for date in dates:
            for time in times:
                building_col = f'{date}_building_shade_percent_at_{time}'
                tree_col = f'{date}_tree_shade_percent_at_{time}'

                try:
                    # Try to build a DataFrame for both columns
                    df_temp = pd.DataFrame({
                        'Shade_Percent': dataset[[building_col, tree_col]].stack(),
                        'Type': dataset[[building_col, tree_col]].stack().index.get_level_values(1).map({
                            building_col: 'Buildings only',
                            tree_col: 'Buildings and Trees'
                        }),
                        'Time': time
                    })
                    plot_data.append(df_temp)

                except KeyError as e:
                    # If a column is missing, print a message and continue
                    print(f"Warning: {e} not found in dataset {dataset_name} for {time}. Skipping this time.")

        if plot_data:
            # Concatenate all dataframes for this dataset
            plot_df_filtered = pd.concat(plot_data, axis=0)

            # Custom palette
            palette = {'Buildings only': '#7f8c8d', 'Buildings and Trees': '#73d016'}  # Grey and Green

            # Mean line properties
            meanlineprops = dict(linestyle='-', linewidth=1.5, color=meanline_color)

            # Boxplot in each subplot
            ax = sns.boxplot(x='Time', y='Shade_Percent', hue='Type', data=plot_df_filtered, palette=palette, fliersize=1.5,
                             flierprops={"marker": "o", "markerfacecolor": "none", "alpha": 0.1, "markeredgecolor": "gray"}, 
                             meanline=True, showmeans=True, meanprops=meanlineprops, ax=axes[i])

            # Titles and labels
            ax.set_title(f"{dataset_name}", fontsize=10)
            ax.set_xlabel('')
            if i == 0:  # Only show the y-axis label on the first plot
                ax.set_ylabel('Shade Percent', fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
            else:
                ax.set_ylabel('')  # No y-label for other subplots

            # Rotate x-ticks
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=7)

            ax.get_legend().remove()

    # Add a single combined legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    mean_line = mlines.Line2D([], [], color=meanline_color, linestyle='-', linewidth=1.5, label='Mean')
    handles.append(mean_line)
    labels.append('Mean')

    # Add combined legend at the bottom
    fig.legend(handles=handles, labels=labels, title='', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=(len(datasets)+1), fontsize=8)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot if a save_path is provided
    if save_path:
        current_date = dt.now().strftime('%Y%m%d')
        filename = f"{current_date}_{osmid}_comparison_plot.png"
        full_save_path = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"Plot saved to {full_save_path}")

    plt.show()



def plot_shade_comparison_rows(datasets, dataset_names, dates, times, osmid, title, save_path=None, meanline_color='orange'):
    """
    Plot a comparison of shade percentages across multiple datasets for specific dates and times.
    The first row of the plot shows 'Buildings only', and the second row shows 'Buildings and Trees'.
    
    Parameters:
    datasets (list): A list of datasets (DataFrames or GeoDataFrames) to compare.
    dataset_names (list): Names of each dataset (for plot titles).
    dates (list): List of date strings in 'YYYYMMDD' format.
    times (list): List of time increments (e.g., ['1130', '1200', '1230']).
    osmid (str): The OSM ID for labeling the plot.
    title (str): Title for the plot.
    save_path (str): Path to save the plot (optional).
    meanline_color (str): Color for the mean line in the plot (default: 'orange').
    """
    num_datasets = len(datasets)
    
    # Prepare a 2-row layout, where the first row is 'Buildings only' and the second is 'Buildings and Trees'
    fig, axes = plt.subplots(2, num_datasets, figsize=(6 * num_datasets, 8), sharey='row')  # Sharing y-axis for each row

    if num_datasets == 1:
        axes = np.array([axes]).T  # Ensure axes is a 2D array for uniform handling even with a single dataset

    # Loop through each dataset and its corresponding name
    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
        plot_data = []

        for date in dates:
            for time in times:
                building_col = f'{date}_building_shade_percent_at_{time}'
                tree_col = f'{date}_tree_shade_percent_at_{time}'

                try:
                    # Try to build a DataFrame for both columns
                    df_temp = pd.DataFrame({
                        'Shade_Percent': dataset[[building_col, tree_col]].stack(),
                        'Type': dataset[[building_col, tree_col]].stack().index.get_level_values(1).map({
                            building_col: 'Buildings only',
                            tree_col: 'Buildings and Trees'
                        }),
                        'Time': time
                    })
                    plot_data.append(df_temp)

                except KeyError as e:
                    # If a column is missing, print a message and continue
                    print(f"Warning: {e} not found in dataset {dataset_name} for {time}. Skipping this time.")

        if plot_data:
            # Concatenate all dataframes for this dataset
            plot_df_filtered = pd.concat(plot_data, axis=0)

            # Custom palette
            palette = {'Buildings only': '#7f8c8d', 'Buildings and Trees': '#73d016'}  # Grey and Green

            # Mean line properties
            meanlineprops = dict(linestyle='-', linewidth=1.5, color=meanline_color)

            # Filter data for 'Buildings only'
            buildings_only_df = plot_df_filtered[plot_df_filtered['Type'] == 'Buildings only']
            trees_df = plot_df_filtered[plot_df_filtered['Type'] == 'Buildings and Trees']

            # Plot 'Buildings only' in the first row
            ax_buildings = sns.boxplot(x='Time', y='Shade_Percent', hue='Type', data=buildings_only_df, palette=[palette['Buildings only']],
                                       fliersize=1.5, flierprops={"marker": "o", "markerfacecolor": "none", "alpha": 0.1, "markeredgecolor": "gray"},
                                       meanline=True, showmeans=True, meanprops=meanlineprops, ax=axes[0, i], legend=False)

            # Plot 'Buildings and Trees' in the second row
            ax_trees = sns.boxplot(x='Time', y='Shade_Percent', hue='Type', data=trees_df, palette=[palette['Buildings and Trees']],
                                   fliersize=1.5, flierprops={"marker": "o", "markerfacecolor": "none", "alpha": 0.1, "markeredgecolor": "gray"},
                                   meanline=True, showmeans=True, meanprops=meanlineprops, ax=axes[1, i], legend=False)

            # Titles and labels
            ax_buildings.set_title(f"{dataset_name}", fontsize=10)
            ax_buildings.set_xlabel('')

            if i == 0:  # Only show the y-axis label on the first column
                ax_buildings.set_ylabel('Shade Percent (Buildings only)', fontsize=8)
                ax_trees.set_ylabel('Shade Percent (Buildings and Trees)', fontsize=8)
            else:
                ax_buildings.set_ylabel('')
                ax_trees.set_ylabel('')

            # Rotate x-ticks safely
            ax_buildings.tick_params(axis='x', rotation=45, labelsize=7)
            ax_trees.tick_params(axis='x', rotation=45, labelsize=7)

            # Ensure no individual legends
            if ax_buildings.get_legend() is not None:
                ax_buildings.get_legend().remove()
            if ax_trees.get_legend() is not None:
                ax_trees.get_legend().remove()

    # Add a single combined legend for both rows
    handles, labels = ax_buildings.get_legend_handles_labels()
    mean_line = mlines.Line2D([], [], color=meanline_color, linestyle='-', linewidth=1.5, label='Mean')
    handles.append(mean_line)
    labels.append('Mean')

    # Add the combined legend below the figure
    fig.legend(handles=handles, labels=labels, title='', loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=(len(datasets)+1), fontsize=8)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot if a save_path is provided
    if save_path:
        current_date = dt.now().strftime('%Y%m%d')
        filename = f"{current_date}_{osmid}_comparison_plot_rows.png"
        full_save_path = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        plt.savefig(full_save_path, bbox_inches='tight')
        print(f"Plot saved to {full_save_path}")

    plt.show()


def plot_shade_distribution(gdf, column_name, cmap='viridis', 
                            theme='dark', zoom=15, save_path=None):
    """
    Visualizes a GeoDataFrame's polygon data on a map with a dark or light background and 
    displays the distribution of the values in a column with the median highlighted.
    
    Parameters:
    - gdf: GeoDataFrame containing polygon data
    - column_name: The name of the column to visualize (must contain numeric data)
    - cmap: Colormap for the map visualization (default 'viridis')
    - theme: 'dark' for dark background (CartoDB DarkMatter) or 'light' for light background (CartoDB Positron)
    - zoom: Zoom level for the basemap (default 15)
    - save_path: If provided, the plot will be saved to the specified file path
    """
    # Choose basemap and styling based on theme
    if theme == 'dark':
        basemap = ctx.providers.CartoDB.DarkMatter
        bg_color = '#333333'
        text_color = 'white'
        spine_color = 'white'
    else:
        basemap = ctx.providers.CartoDB.Positron
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Reproject to Web Mercator (EPSG:3857) for basemap alignment
    gdf = gdf.to_crs(epsg=3857)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [4, 1]})

    # 1. Map Visualization with Chosen Background
    gdf.plot(
        column=column_name, 
        cmap=cmap, 
        legend=False,  # Removed the legend
        ax=ax[0],
        edgecolor='face',
    )

    # Add basemap (from contextily)
    ctx.add_basemap(
        ax[0], 
        source=basemap, 
        zoom=zoom
    )

    # Remove axis for the map
    ax[0].axis('off')
    ax[0].set_title(f'{column_name.replace("_", " ").capitalize()} Map', fontsize=14, color=text_color)

    # 2. Distribution Graph with Highlighted Median
    sns.kdeplot(
        gdf[column_name], 
        color='lightgreen', 
        ax=ax[1]
    )  # Show a smooth line for the distribution

    # Highlight median
    median_val = gdf[column_name].median()
    ax[1].axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2f}%')

    # Formatting the distribution plot
    ax[1].set_xlabel(f'{column_name.replace("_", " ").capitalize()} (%)', fontsize=12, color=text_color)
    ax[1].set_ylabel('')
    ax[1].set_yticks([])  # Remove y-axis labels
    ax[1].legend()

    # Set background and text color for the distribution plot
    ax[1].set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax[1].spines['left'].set_color(bg_color)
    ax[1].spines['right'].set_color(bg_color)
    ax[1].spines['top'].set_color(bg_color)
    ax[1].spines['bottom'].set_color(spine_color)

    # Update tick and label colors for readability
    ax[1].tick_params(colors=text_color)
    ax[1].xaxis.label.set_color(text_color)
    ax[1].legend(facecolor=bg_color, edgecolor=spine_color)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')

    # Display the plot
    plt.show()
    
# Usage example:
# plot_shade_distribution(gdf, 'tree_shade_percent_day_average', theme='light', save_path='shade_plot.png')

def plot_hexmap_shade_distribution(gdf, column_name, cmap='viridis', 
                                   theme='dark', zoom=15, save_path=None):
    """
    Visualizes a hexmap of a GeoDataFrame's polygon data on a map with a dark or light background and 
    displays the distribution of the values in a column with the median highlighted.
    
    Parameters:
    - gdf: GeoDataFrame containing polygon data
    - column_name: The name of the column to visualize (must contain numeric data)
    - cmap: Colormap for the hexmap visualization (default 'viridis')
    - theme: 'dark' for dark background (CartoDB DarkMatter) or 'light' for light background (CartoDB Positron)
    - zoom: Zoom level for the basemap (default 15)
    - save_path: If provided, the plot will be saved to the specified file path
    """
    # Choose basemap and styling based on theme
    if theme == 'dark':
        basemap = ctx.providers.CartoDB.DarkMatter
        bg_color = '#333333'
        text_color = 'white'
        spine_color = 'white'
    else:
        basemap = ctx.providers.CartoDB.Positron
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Reproject to Web Mercator (EPSG:3857) for basemap alignment
    gdf = gdf.to_crs(epsg=3857)

    # Calculate centroids of each polygon for hexbin plotting
    gdf['centroid'] = gdf.centroid

    # Extract x and y coordinates of the centroids
    x = gdf['centroid'].x
    y = gdf['centroid'].y
    values = gdf[column_name]

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [4, 1]})

    # 1. Hexmap Visualization with Chosen Background
    hb = ax[0].hexbin(x, y, C=values, gridsize=50, cmap=cmap, reduce_C_function=np.mean)

    # Add basemap (from contextily)
    ctx.add_basemap(
        ax[0], 
        source=basemap, 
        zoom=zoom
    )

    # Remove axis for the map
    ax[0].axis('off')
    ax[0].set_title(f'{column_name.replace("_", " ").capitalize()} Hexmap', fontsize=14, color=text_color)

    # Add color bar to show scale of values
    cb = plt.colorbar(hb, ax=ax[0])
    cb.set_label(f'{column_name.replace("_", " ").capitalize()} (%)')

    # 2. Distribution Graph with Highlighted Median
    sns.kdeplot(
        gdf[column_name], 
        color='lightgreen', 
        ax=ax[1]
    )  # Show a smooth line for the distribution

    # Highlight median
    median_val = gdf[column_name].median()
    ax[1].axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2f}%')

    # Formatting the distribution plot
    ax[1].set_xlabel(f'{column_name.replace("_", " ").capitalize()} (%)', fontsize=12, color=text_color)
    ax[1].set_ylabel('')
    ax[1].set_yticks([])  # Remove y-axis labels
    ax[1].legend()

    # Set background and text color for the distribution plot
    ax[1].set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax[1].spines['left'].set_color(bg_color)
    ax[1].spines['right'].set_color(bg_color)
    ax[1].spines['top'].set_color(bg_color)
    ax[1].spines['bottom'].set_color(spine_color)

    # Update tick and label colors for readability
    ax[1].tick_params(colors=text_color)
    ax[1].xaxis.label.set_color(text_color)
    ax[1].legend(facecolor=bg_color, edgecolor=spine_color)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')

    # Display the plot
    plt.show()

# Usage example:
# plot_hexmap_shade_distribution(gdf, 'tree_shade_percent_day_average', theme='light', save_path='hexmap_shade_plot.png')



# def plot_shade_statistics_per_city(df_dict, cities, time_increments, column_prefix='20240620_tree_shade_percent_at_', title='Shade Percentage Over Time (Mean & Median)', save_path=None, file_name=None):
#     """
#     Plot the mean and median shade percentages over time for multiple cities.
#     Each subplot shows one metric (mean or median), with multiple cities plotted on the same subplot.
    
#     Parameters:
#     - df_dict: Dictionary containing DataFrames for each city, keyed by city name.
#     - cities: List of city names corresponding to the keys in df_dict.
#     - time_increments: List of time increments to plot.
#     - column_prefix: Prefix for the column names containing shade percentages (default is 'tree_shade_percent_at_').
#     - title: Title of the plot.
#     - save_path: Directory path where to save the plot (optional).
#     - file_name: Name for the saved plot file (optional).
    
#     Returns:
#     - None
#     """
#     n_metrics = 2  # We are plotting mean and median
#     fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 10), squeeze=False)  # Two rows: one for average, one for median
#     axes = axes.flatten()

#     colors = sns.color_palette("husl", len(cities))  # Assign a unique color for each city
    
#     lines = []
#     labels = []

#     for idx, city in enumerate(cities):
#         city_df = df_dict[city]  # Get the DataFrame for the current city
#         color = colors[idx]  # Use the color assigned to this city

#         mean_shade_per_time = {}
#         median_shade_per_time = {}

#         for time in time_increments:
#             column_name = f'{column_prefix}{time}'
#             if column_name in city_df.columns:
#                 mean_shade_per_time[time] = city_df[column_name].mean()
#                 median_shade_per_time[time] = city_df[column_name].median()
#             else:
#                 mean_shade_per_time[time] = 100
#                 median_shade_per_time[time] = 100

#         # Convert to DataFrames for easier plotting
#         mean_shade_df = pd.DataFrame(list(mean_shade_per_time.items()), columns=['Time', 'Mean Shade Percentage'])
#         median_shade_df = pd.DataFrame(list(median_shade_per_time.items()), columns=['Time', 'Median Shade Percentage'])

#         # Plot the mean for each city in the first row
#         line_avg, = axes[0].plot(mean_shade_df['Time'], mean_shade_df['Mean Shade Percentage'], marker='o', markersize=3, linewidth=1.2, color=color, label=city)

#         # Plot the median for each city in the second row
#         line_med, = axes[1].plot(median_shade_df['Time'], median_shade_df['Median Shade Percentage'], marker='x', markersize=3, linestyle='--', linewidth=1.2, color=color)

#         # # Keep track of lines and labels for the legend
#         # if idx == 0:
#         #     lines.append(line_avg)
#         #     labels.append(city)
#         lines.append(line_avg)
#         labels.append(city)

#     # Customize the mean subplot
#     axes[0].set_ylim(bottom=0)
#     axes[0].axhline(30, color='red', linestyle='dashed', linewidth=0.8, label='Minimum Threshold')
#     axes[0].set_ylabel('Mean Shade (%)', fontsize=12)
#     axes[0].set_title('Mean Shade Percentage', fontsize=14)
#     axes[0].grid(True)
#     axes[0].set_xticks(range(0, len(time_increments), 4))
#     axes[0].set_xticklabels(time_increments[::4], rotation=45, fontsize=10)
#     axes[0].tick_params(axis='y', labelsize=10)

#     # Customize the median subplot
#     axes[1].set_ylim(bottom=0)
#     axes[1].axhline(30, color='red', linestyle='dashed', linewidth=0.8)
#     axes[1].set_ylabel('Median Shade (%)', fontsize=12)
#     axes[1].set_title('Median Shade Percentage', fontsize=14)
#     axes[1].grid(True)
#     axes[1].set_xticks(range(0, len(time_increments), 4))
#     axes[1].set_xticklabels(time_increments[::4], rotation=45, fontsize=10)
#     axes[1].tick_params(axis='y', labelsize=10)

#     # Add a single shared legend for the cities, outside of the plot
#     fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=len(cities), fontsize=10)

#     plt.suptitle(title, fontsize=16)  # Main title
#     plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to prevent overlap

#     # Save the plot if the path and file name are provided
#     if save_path and file_name:
#         os.makedirs(save_path, exist_ok=True)
#         full_path = os.path.join(save_path, f"{file_name}.png")
#         plt.savefig(full_path, bbox_inches='tight', facecolor=fig.get_facecolor())
#         print(f"Plot saved at {full_path}")

#     plt.show()

# Example usage:
# Replace 'df_dict' with your actual dictionary of DataFrames for each city
# plot_shade_statistics_per_city(df_dict, cities, time_increments, title='Shade Percentages for Different Cities')



def plot_shade_statistics_per_city(df_dict, cities, time_increments, column_prefix_1='20240620_tree_shade_percent_at_', column_prefix_2=None, title='Shade Percentage Over Time (Mean & Median)', save_path=None, file_name=None):
    """
    Plot the mean and median shade percentages over time for multiple cities.
    Each subplot shows one metric (mean or median), with multiple cities plotted on the same subplot.
    
    Parameters:
    - df_dict: Dictionary containing DataFrames for each city, keyed by city name.
    - cities: List of city names corresponding to the keys in df_dict.
    - time_increments: List of time increments to plot.
    - column_prefix_1: Prefix for the first set of shade percentage columns (default is 'tree_shade_percent_at_').
    - column_prefix_2: Prefix for the second set of shade percentage columns (optional, if None, only one set will be plotted).
    - title: Title of the plot.
    - save_path: Directory path where to save the plot (optional).
    - file_name: Name for the saved plot file (optional).
    
    Returns:
    - None
    """
    n_metrics = 2  # We are plotting mean and median
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 10), squeeze=False)  # Two rows: one for average, one for median
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(cities))  # Assign a unique color for each city
    
    lines = []
    labels = []

    for idx, city in enumerate(cities):
        city_df = df_dict[city]  # Get the DataFrame for the current city
        color = colors[idx]  # Use the color assigned to this city

        mean_shade_per_time_1 = {}
        median_shade_per_time_1 = {}
        mean_shade_per_time_2 = {}
        median_shade_per_time_2 = {}

        for time in time_increments:
            column_name_1 = f'{column_prefix_1}{time}'
            column_name_2 = f'{column_prefix_2}{time}' if column_prefix_2 else None
            
            if column_name_1 in city_df.columns:
                mean_shade_per_time_1[time] = city_df[column_name_1].mean()
                median_shade_per_time_1[time] = city_df[column_name_1].median()
            else:
                mean_shade_per_time_1[time] = 100
                median_shade_per_time_1[time] = 100

            if column_name_2 and column_name_2 in city_df.columns:
                mean_shade_per_time_2[time] = city_df[column_name_2].mean()
                median_shade_per_time_2[time] = city_df[column_name_2].median()
            else:
                mean_shade_per_time_2[time] = 100
                median_shade_per_time_2[time] = 100

        # Convert to DataFrames for easier plotting
        mean_shade_df_1 = pd.DataFrame(list(mean_shade_per_time_1.items()), columns=['Time', 'Mean Shade Percentage 1'])
        median_shade_df_1 = pd.DataFrame(list(median_shade_per_time_1.items()), columns=['Time', 'Median Shade Percentage 1'])

        if column_prefix_2:
            mean_shade_df_2 = pd.DataFrame(list(mean_shade_per_time_2.items()), columns=['Time', 'Mean Shade Percentage 2'])
            median_shade_df_2 = pd.DataFrame(list(median_shade_per_time_2.items()), columns=['Time', 'Median Shade Percentage 2'])

        # Plot the mean for each city in the first row
        line_avg_1, = axes[0].plot(mean_shade_df_1['Time'], mean_shade_df_1['Mean Shade Percentage 1'], marker='o', markersize=3, linewidth=1.2, color=color, label=f'{city} - 1')

        # Plot the median for each city in the second row
        line_med_1, = axes[1].plot(median_shade_df_1['Time'], median_shade_df_1['Median Shade Percentage 1'], marker='x', markersize=3, linestyle='--', linewidth=1.2, color=color, label=f'{city} - 1')

        # If a second column prefix is provided, plot those as well
        if column_prefix_2:
            line_avg_2, = axes[0].plot(mean_shade_df_2['Time'], mean_shade_df_2['Mean Shade Percentage 2'], marker='o', markersize=3, linewidth=1.2, color=color, linestyle=':', label=f'{city} - 2')
            line_med_2, = axes[1].plot(
                median_shade_df_2['Time'], 
                median_shade_df_2['Median Shade Percentage 2'], 
                marker='x', 
                markersize=3, 
                linewidth=1.2, 
                color=color, 
                linestyle=':',  # Keep only this linestyle
                label=f'{city} - 2'
            )

        lines.append(line_avg_1)
        labels.append(f'{city} - 1')
        
        if column_prefix_2:
            lines.append(line_avg_2)
            labels.append(f'{city} - 2')

    # Customize the mean subplot
    axes[0].set_ylim(bottom=0)
    axes[0].axhline(30, color='red', linestyle='dashed', linewidth=0.8, label='Minimum Threshold')
    axes[0].set_ylabel('Mean Shade (%)', fontsize=12)
    axes[0].set_title('Mean Shade Percentage', fontsize=14)
    axes[0].grid(True)
    axes[0].set_xticks(range(0, len(time_increments), 4))
    axes[0].set_xticklabels(time_increments[::4], rotation=45, fontsize=10)
    axes[0].tick_params(axis='y', labelsize=10)

    # Customize the median subplot
    axes[1].set_ylim(bottom=0)
    axes[1].axhline(30, color='red', linestyle='dashed', linewidth=0.8)
    axes[1].set_ylabel('Median Shade (%)', fontsize=12)
    axes[1].set_title('Median Shade Percentage', fontsize=14)
    axes[1].grid(True)
    axes[1].set_xticks(range(0, len(time_increments), 4))
    axes[1].set_xticklabels(time_increments[::4], rotation=45, fontsize=10)
    axes[1].tick_params(axis='y', labelsize=10)

    # Add a single shared legend for the cities, outside of the plot
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=len(labels), fontsize=10)

    plt.suptitle(title, fontsize=16)  # Main title
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to prevent overlap

    # Save the plot if the path and file name are provided
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{file_name}.png")
        plt.savefig(full_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved at {full_path}")

    plt.show()