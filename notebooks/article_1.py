#!/usr/bin/env python
# coding: utf-8

# # Data Insight: Top Factors For Startup Success
#
# Learn about the **factors** leading to startups being acquired and **how** to apply exploratory data analysis to identify these factors.
#
# We will explore the [Startup Success Prediction dataset](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction) released on [Kaggle](https://www.kaggle.com/). It contains information about industries, acquisitions, and investment details of nearly a thousand startups based in the USA from the 1980s to 2013.
#
# It's essential to understand how to transform raw data into meaningful information. To achieve this, we will follow the workflow below:
#
# 1. Data exploration
# 2. Pre-processing
# 3. Data analysis
# 4. Data-driven conclusions
#
#
# Here are the main objectives of our analysis:
#
# - Examine the industries where startup acquisitions are most common, highlighting how crucial the type of industry is.
# - Examine whether reaching certain milestones increases a startup's likelihood of acquisition.
# - Analyze the average total funding of startups that get acquired versus those that close.
# - Evaluate if startups with Series A, B, or C funding are more likely to succeed than those backed by angel investors or VC.
#
# Before we start, let's investigate the startup related concepts to understand the key challenges in this ecosystem.
#
# ## Understanding the Startup Ecosystem
# In the startup field, there are important concepts to understand. First, startups raise money in stages called **funding rounds**, like Series A, B, or C. These rounds enables the startups to invest in the necessary resources for their growth. Second, startups set goals called **milestones**, like launching a product or getting a certain number of users. These reflect their performance to the investors.
#
# Finally, startups secure funding from to types of source:
# - **Angel investors** are high net worth individuals who provide personal capital for investment. - **venture capital firms** are corporate entities specializing in investing in startups.
#
# Let’s get in to the preliminary steps.

# ## Data Exploration

# Data exploration serves as the first step in data analysis, involving the examination of datasets to identify patterns and detect anomalies.
#
# The following libraries provide the necessary tools and functions for data manipulation, analysis, and visualization.

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore


# Now, we load the data into a [Pandas](https://pandas.pydata.org/) DataFrame.

# In[3]:


file_path = os.path.join('..', 'saved_model', 'startup.csv')
data_df = pd.read_csv(file_path, encoding="ISO-8859-1")


# In[4]:


data_df.head(2)


# Next, let's focus on understanding the significance of the Target Variable.

# The `status` variable is the most important feature in this dataset. This binary variable indicates whether a startup has been acquired or closed.
#
# When analyzing the variable, it shows that 597 startups in our dataset were acquired, while 326 startups closed down.

# In[5]:


data_df['status'].head(2)


# ## Data Pre-processing

# The pre-processing step involves cleaning and organizing the raw data.
#
# We'll accomplish following these steps:
#
# - The **Data cleaning** step consists on identifying and correcting data errors, such as typos, duplicates, or incorrect entries. It is essential for maintaining data integrity. We'll focus on handling:
# - - The irrelevant features
# - - The missing values
# - - The negative values where it shouldn't be so.
#
# - The **Data transformation** step modifies data to re-organize them into a format appropriate for performing analysis. These are the common transformatio that we will perform:
# - - **Normalization** and **standardization**: These techniques adjust data to a common scale, making it easier to compare them.
# - - **Encoding categorical data**: Transforming categorical variables into numerical formats.

# ### Handle Irrelevant Features

# Irrelevant feature refers to a data attribute or variable that does not provide valuable or meaningful information for the task at hand. Here are some reasons why removing them is beneficial:
#
# - Eliminating irrelevant variables simplifies the analysis process.
# - It ensures focus on meaningful factors by reducing clutter and improving clarity in data exploration.
#
# To accomplish this, we will implement the following method.

# In[6]:


data_df = data_df.drop([
    'Unnamed: 0',   #Irrelevant data
    'latitude',     #Irrelevant data
    'longitude',    #Irrelevant data
    'zip_code',     #Irrelevant data
    'id',           #Irrelevant data
    'Unnamed: 6',   #Irrelevant data
    'name',         #Irrelevant data
    'labels',       #Same data as status
    'state_code.1', #Same data as state_code
    'is_CA',        #Binary column, not needed
    'is_NY',        #Binary column, not needed
    'is_MA',        #Binary column, not needed
    'is_TX',        #Binary column, not needed
    'is_otherstate',#Binary column, not needed
    'is_software',  #Binary column, not needed
    'is_web',       #Binary column, not needed
    'is_mobile',    #Binary column, not needed
    'is_enterprise',#Binary column, not needed
    'is_advertising',#Binary column, not needed
    'is_gamesvideo',#Binary column, not needed
    'is_ecommerce', #Binary column, not needed
    'is_biotech',   #Binary column, not needed
    'is_consulting',#Binary column, not needed
    'is_othercategory',#Binary column, not needed
    'object_id'],   #Irrelevant data
    axis=1,
).copy()
data_df.head(2)


#
# After discarding irrelevant variables, we've simplified the DataFrame. From 50 columns initially, it now contains 24 columns. Let's look at the data.

# In[7]:


data_df.info()


# A preliminary review allows us to understand the nature of the data, especially focusing on identifying key data types that will help us to make decisions.
#
# Before starting to do any data cleaning. Let's begin with an overview of some categorical variables such as `category_code` and `status` that will help us to gain valuable insights.
#
# To achieve this, we will utilize **bar chart** visualizations, known for their effectiveness in displaying data

# ### Top 10 Sectors with High Startup Acquisition Rates

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Getting the order of categories based on their frequency
# (now for y-axis since we are doing a horizontal bar plot)
order = data_df["category_code"].value_counts().index[:10]

# Create the countplot with horizontal bars
plt.figure(figsize=(18, 9), dpi=300)

# Specify the hue order explicitly
ax = sns.countplot(
    y="category_code", data=data_df, order=order, palette="pastel",
    hue='status', hue_order=['acquired', 'closed']
)

plt.title("Top 10 Industries Where Startups Are Frequently Acquired", fontsize=22, pad=20, fontweight='bold')
plt.xlabel("Number of Startups",fontsize=17)
plt.ylabel("Category Code", fontsize=17)
plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

# Set the font size of the category labels using tick_params
ax.tick_params(axis='y', labelsize=14)

# Iterate through the patches (bars) and add text
for bar in ax.patches:
    # Get the position and width of the bar
    bar_y = bar.get_y() + bar.get_height() / 2  # Y-coordinate of the center of the bar
    bar_width = bar.get_width()  # Width of the bar (count value)

    # Add text to the right end of the bar
    ax.text(bar_width, bar_y, f'{int(bar_width)}',
            fontsize=12, color='black', ha='left', va='center')

plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
plt.show()


# As observed in the bar chart, we identify the following:
#
# - The software industry not only has the highest number of startups but also sees a significant portion of these ventures being acquired by larger entities.
#
# - This trend highlights how the software industry's constant innovation and growth attracts big companies interested in new technologies or looking to grow their tech capabilities.
#
# - Similarly, the web, mobile, and enterprise sectors emerge as vibrant ecosystems, containing many startups.
#
# <!---
# Rephrase
# -->
# Now that we know the top industries for startups, our next task is to examine time related factors. This will help us understand how time affects trends in startup acquisitions.

# ### Year of the highest number of startups founded

# In order to address this question, it will be necessary to generate an additional variable named `proportions`.
#
# Calculating the number of startups founded each year and comparing it to the total number of startups gives us a normalized view of the data.

# In[8]:


# Convert 'founded_at' column to datetime objects
data_df['founded_at'] = pd.to_datetime(data_df['founded_at'])

# Extract and format the year from 'founded_at' as 'founded_year'
data_df['founded_year'] = data_df['founded_at'].dt.strftime('%Y')

# Group the DataFrame by 'founded_year' and count occurrences
prop_df = data_df.groupby('founded_year').size().reset_index(name = 'counts')

# Calculate the proportions of startups founded each year
prop_df['proportions'] = prop_df['counts']/prop_df['counts'].sum()


# In[9]:


fig, ax = plt.subplots(figsize=(50, 20))
# Adjust the font size as needed
ax.tick_params(axis='x', labelsize=35)

# Optionally, change the font size of the tick labels on the y-axis as well
ax.tick_params(axis='y', labelsize=35)

ax.set_xlabel('Founded Year', fontsize=40,labelpad=45)  # Change the font size as needed
ax.set_ylabel('Proportions', fontsize=40, labelpad=45)
# Create a barplot without a hue parameter
barplot = sns.barplot(data=prop_df, x='founded_year', y='proportions')

# Manually set colors for each bar if you want them to be different
colors = sns.color_palette("rainbow", n_colors=len(prop_df['founded_year']))
for i, bar in enumerate(barplot.patches):
    bar.set_color(colors[i % len(colors)])

plt.title('Distribution of Number of Startups Over Years (Acquired/Not)',fontsize=55, pad=70, fontweight='bold')
plt.show()


# Based on the visualization, we identify the following:
#
# - Looking at the history of startups, we see a significant increase in activity up until 2007, indicating an era of rapid growth. The rise in startup activity during this era is probably connected to the rise of the internet and digital technologies.
#
# - However, after 2007, there was a drop that could have been caused by the worldwide financial crisis, market saturation, economic declines, or changes in how investors act.
#
# Let's go a head to explore how location influences startup success.

# ### Startup Acquisitions Across U.S. States

# Let's proceed to create another plot that will help us identify the U.S. states with the highest rates of startup acquisitions.

# In[10]:


import matplotlib.pyplot as plt

# Assuming 'data_df' is your DataFrame and it contains 'state_code' and 'status' columns
grouped_data = data_df.groupby(['state_code', 'status']).size().unstack().fillna(0)

# Sort the DataFrame by the sum of 'acquired' and 'closed' for each state and take the first 10
grouped_data = grouped_data.sort_values(by=['acquired', 'closed'], ascending=False).head(10)

# Convert the index to string to ensure proper handling as categorical data
grouped_data.index = grouped_data.index.astype(str)

# Create the stacked bar chart
ax = grouped_data.plot(kind='bar', stacked=True, figsize=(15, 8), color=['skyblue', 'salmon'])
plt.title('Number of Acquired vs. Closed Startups in Top 10 States',fontsize=20, pad=25, fontweight='bold')
plt.xlabel('State Code', fontsize=18,labelpad=10)
plt.ylabel('Number of Startups', fontsize=18,labelpad=20)
plt.xticks(rotation=45)  # Rotates the state labels for better readability
plt.legend(title='Status')

# Define a threshold for displaying annotations
threshold = 50  # Adjust the threshold as needed

# Add text annotations
for i, (state_code, row) in enumerate(grouped_data.iterrows()):
    cumulative_height = 0
    for status in ['acquired', 'closed']:  # Iterate in a specific order
        count = row[status]
        # Display annotation only if count is above the threshold
        if count > threshold:
            # Position text at the center of each segment
            ax.text(i, cumulative_height + count / 2, f'{int(count):,}',
                    fontsize=8, color='black', ha='center', va='center')
        cumulative_height += count

# Slightly adjust the ylim to make room for text annotations
ax.set_ylim(0, grouped_data.max().sum() * 1.1)  # Increase the y-axis limit
plt.show()


# The bar chart shows the following location trends:
#
# - California (CA) stands out with the highest total number of startups, where the number of acquired startups is more than double the number closed.
#
# - New York (NY) and Massachusetts (MA) also exhibit a pattern where more startups are being acquired than closed, suggesting a favorable startup environment in these states as well.
#
# - In other states such as Washington (WA), Texas (TX), and Colorado (CO), the numbers of acquisitions and closures are closer, pointing to a more challenging environment.

# We've seen how important visuals are for understanding data and identifying important discoveries.
#
# After using the categorical `status` variable, we'll refine it by converting `acquired` and `closed` into numeric values to facilitate analysis.
#
# To accomplish this, we'll use the **mapping method**, which assigns a unique numerical value to each category.

# In[11]:


data_df['status'] = data_df.status.map({'acquired':1, 'closed':0})
data_df.head(2)


#  ### Missing Values

# Missing values (NA) are data points in a dataset where information is simply not available or recorded for a specific attribute or variable.
#
# Importance of **handling** missing values:
#
# - Preserve the integrity of your dataset. Without proper handling, NA can lead to biased or misleading results in statistical analyses.
#
# - Maintain the reliability of your analyses, making them more robust and trustworthy.

# In[12]:


x= data_df.isnull().sum()
x.sort_values(ascending=False)


# It's important to remember that handling NA varies based on data characteristics and analysis goals. The chosen method should be tailored to each case.
#
# As shown above, we've identified three features that contain missing values.
#
# Given that `age_first_milestone_year` and `age_last_milestone_year` are continuous numerical variables, our chosen imputation method will involve filling the missing values with the respective means for these variables.

# **Fillna() Method**

# The method provides:
#
# - Flexibility in handling missing data by replacing NaN values with custom alternatives, like specific numbers, text, or computed values.
#
#
# This is a simple and easily applicable method. By incorporating values centered around the data's central tendency, it retains the original distribution. It's especially well suited for continuous numerical variables.
#

# In[13]:


data_df["age_first_milestone_year"] = data_df["age_first_milestone_year"].fillna(data_df["age_first_milestone_year"].mean())
data_df["age_last_milestone_year"] = data_df["age_last_milestone_year"].fillna(data_df["age_last_milestone_year"].mean())


# In[14]:


data= data_df.isnull().sum()
data.sort_values(ascending=False)


# **General Changes:**
#
# - Imputed values for `age_first_milestone_year` and `age_last_milestone_year`.
#
# - Missing values in `closed_at` represent startups that are still open and will not be filled for now. This approach preserves the information that certain startups are still active without arbitrarily assigning a closure date.
#
# Now that we've established the significance of handling missing values, let's proceed with the next step.

# ### Negative values

# Negative values can sometimes offer valuable insights; for example, in financial data, they suggest debt or losses. Removing them without careful consideration could result in missing critical information. However, in other contexts, negative numbers might indicate errors or require changes to the data.
#
# Before we continue, we will convert the following columns into datetime format to ensure that they are suitable for time based analysis.

# In[15]:


data_df.founded_at=pd.to_datetime(data_df.founded_at)
data_df.first_funding_at=pd.to_datetime(data_df.first_funding_at)
data_df.last_funding_at=pd.to_datetime(data_df.last_funding_at)


# #### Identifying Negative Values: Box Plot

# To find negative values, we'll use **box plots**. Box plots make it easy to spot and fix negative values that could be mistakes or unusual data.
#
# According to our data set, most of the features are binary which are not suitable for negative values detection, therefore we will consider only the following continuous variables.

# In[16]:


featuresNumfinal = ['avg_participants','age_first_funding_year','age_last_funding_year','age_first_milestone_year','age_last_milestone_year']
plt.figure(figsize=(15, 7))
for i in range(0, len(featuresNumfinal)):
    plt.subplot(1, len(featuresNumfinal), i+1)
    sns.boxplot(y=data_df[featuresNumfinal[i]], color='blue', orient='v')
    plt.tight_layout()


# The box plots show us these main observations:
#
# - In box plots 2, 3, 4, and 5, the points that appear outside the main body of the box plot represent values that are outside the typical range of the dataset.
#
# - The points lying above the upper whisker represent the positive outliers. We will address these in a later stage of our analysis.
#
# Let's explore another visualization that will help us clearly identify negative values within our dataset.

# #### Identifying Negative Values: Scatterplot

# After using box plots to find negative values in each variable, let's now look at how two variables relate to each other and spot negative values using scatter plots.
#
# **Scatter plots** help us see the connection between two variables, making it easier to find unusual points or negative values when we look at them together.

# In[17]:


def outliers_analysis(data_df):
    # Perform calculations and add results as new columns in the DataFrame
    data_df['diff_founded_first_funding'] = data_df["first_funding_at"].dt.year - data_df["founded_at"].dt.year
    data_df['diff_founded_last_funding'] = data_df["last_funding_at"].dt.year - data_df["founded_at"].dt.year

    # Define a custom color palette for negative and non-negative values
    custom_palette = {0: "blue", 1: "red"}

    # Create new columns to identify negative values
    data_df['negative_first_funding'] = (data_df["age_first_funding_year"] < 0).astype(int)
    data_df['negative_last_funding'] = (data_df["age_last_funding_year"] < 0).astype(int)
    data_df['negative_first_milestone'] = (data_df["age_first_milestone_year"] < 0).astype(int)
    data_df['negative_last_milestone'] = (data_df["age_last_milestone_year"] < 0).astype(int)

    plt.figure(figsize=(18, 3), dpi=100)

    # Configuration for each subplot
    plots_config = [
        {
            "x": "diff_founded_first_funding",
            "y": "age_first_funding_year",
            "hue": "negative_first_funding",
            "xlabel": "Difference 'Founded' and 'First Funding'",
            "ylabel": "Age at First Funding Year",
            "data": data_df  # Specify the DataFrame
        },
        {
            "x": "diff_founded_last_funding",
            "y": "age_last_funding_year",
            "hue": "negative_last_funding",
            "xlabel": "Difference 'Founded' and 'Last Funding'",
            "ylabel": "Age at Last Funding Year",
            "data": data_df
        },
        {
            "x": "age_first_funding_year",
            "y": "age_first_milestone_year",
            "hue": "negative_first_milestone",
            "xlabel": "Age at First Funding Year",
            "ylabel": "Age at First Milestone Year",
            "data": data_df
        },
        {
            "x": "age_last_funding_year",
            "y": "age_last_milestone_year",
            "hue": "negative_last_milestone",
            "xlabel": "Age at Last Funding Year",
            "ylabel": "Age at Last Milestone Year",
            "data": data_df
        }
    ]

    # Generate each scatter plot based on the configuration
    for i, config in enumerate(plots_config, start=1):
        plt.subplot(1, 4, i)
        sns.scatterplot(
            x=config["x"],
            y=config["y"],
            hue=config["hue"],
            palette=custom_palette,
            data=config["data"]  # Pass the DataFrame here
        )
        plt.xlabel(config["xlabel"])
        plt.ylabel(config["ylabel"])

    plt.tight_layout()  # Adjust the layout
    plt.show()

outliers_analysis(data_df)


# From the **scatterplot** we identified the following:
#
# - Negative values are highlighted in red, while positive values are indicated in blue.
#
# - The occurrence of negative values could be attributed to several reasons, such a data entry errors, or a relative timing. However, without definitive context, we cannot be certain of the exact cause.
#
# For the purposes of our analysis, we'll focus on how to handle these negative values. By doing so, we can maintain the integrity of our analysis and ensure that our insights are based on accurate and representative data.
#
# Let's identify which column specifically contains negative values.

# In[18]:


age_columns = ["age_first_funding_year", "age_last_funding_year", "age_first_milestone_year", "age_last_milestone_year"]
for column in age_columns:
    has_negative_values = (data_df[column] < 0).any()
    print(f"Negative values in '{column}' column: {has_negative_values}")


# Once we've identified them, we can proceed to address them.

# #### Handle Negative Values: Function np.abs()

# This function is a valuable tool in data manipulation and numerical analysis for quickly and reliably obtaining absolute values.
#
# By taking the **absolute values** (converting them to positive values), you remove the negative sign and focus on the magnitude or distance from zero.

# In[19]:


data_df["age_first_funding_year"]=np.abs(data_df["age_first_funding_year"])
data_df["age_last_funding_year"]=np.abs(data_df["age_last_funding_year"])
data_df["age_first_milestone_year"]=np.abs(data_df["age_first_milestone_year"])
data_df["age_last_milestone_year"]=np.abs(data_df["age_last_milestone_year"])


# In[20]:


age_columns = ["age_first_funding_year", "age_last_funding_year", "age_first_milestone_year", "age_last_milestone_year"]
for column in age_columns:
    has_negative_values = (data_df[column] < 0).any()
    print(f"Negative values in '{column}' column: {has_negative_values}")


#  In outlier detection or anomaly detection tasks, absolute values can help identify extreme deviations regardless of their direction.
#
# Now, let's revisit our plot to observe the changes that occurred after the removal of negative values.

# In[21]:


def identify_outliers(data_df):
    threshold = 4  # Example threshold
    column_name = "age_first_funding_year"
    column = data_df[column_name]
    mean = column.mean()
    std = column.std()
    is_outlier = (np.abs(column - mean) > threshold * std)
    data_df['is_outlier'] = is_outlier.astype(int)
    return data_df

data_df = identify_outliers(data_df)

def plot_outlier_scatterplots(data_df):
    custom_palette = {0: "blue", 1: "red"}
    plt.figure(figsize=(16, 3), dpi=100)

    # Plot configurations
    plot_configs = [
        {
            "x": np.abs(data_df["first_funding_at"].dt.year - data_df["founded_at"].dt.year),
            "y": data_df["age_first_funding_year"],
            "label": "Difference 'Founded' and 'First Funding'"
        },
        {
            "x": np.abs(data_df["last_funding_at"].dt.year - data_df["founded_at"].dt.year),
            "y": data_df["age_last_funding_year"],
            "label": "Difference 'Founded' and 'Last Funding'"
        },
        {
            "x": data_df["age_first_funding_year"],
            "y": data_df["age_first_milestone_year"],
            "label": None  # No label specified
        },
        {
            "x": data_df["age_last_funding_year"],
            "y": data_df["age_last_milestone_year"],
            "label": None  # No label specified
        }
    ]

    # Create each scatter plot
    for i, config in enumerate(plot_configs, start=1):
        plt.subplot(1, 4, i)
        sns.scatterplot(x=config["x"], y=config["y"], hue=data_df["is_outlier"], palette=custom_palette)
        if config["label"]:
            plt.xlabel(config["label"])

    plt.tight_layout()  # Adjust the layout
    plt.show()

plot_outlier_scatterplots(data_df)


# The outcome was a cleaner dataset, with all negative values successfully removed.
#
# This time, we focus on the positive outliers (the red points) that were identified using the **z-score** method in our visualization.
#
#  Let's proceed to analyze this method in detail.

#  ### Outliers

# Outliers are data points that significantly deviate from the main group of values in a dataset. These extreme values can distort statistical analyses and impact the reliability of our findings.
# Earlier, we use methods such as Box Plot and Scatter Plot.
#
# We will delve into other two methods to identify the outliers:
#
# - **Histograms** to observe data distribution and spot anomalies.
#
# - Statistical methods, including the calculation of **Z-scores**.

# #### Identifying Outliers: Histogram Analysis

# By closely examining the histogram, we can spot any data points that fall far outside the typical range of values. These outliers often appear as isolated bars or spikes on the histogram, indicating their deviation from the majority of the data.
#
# Let's determine which variables have the highest number of outliers.

# In[22]:


# Select only numeric columns for histogram plots
specific_variables = [
    "age_first_funding_year", "age_last_funding_year",
    "age_first_milestone_year", "age_last_milestone_year",
    "funding_total_usd", "funding_rounds",
    "milestones", "relationships"
]

# Determine the number of rows and columns for the subplots
num_variables = len(specific_variables)
num_columns = 5  # You can choose how many columns you want per row
num_rows = int(np.ceil(num_variables / num_columns))

# Set the overall figure size to give each subplot a little more room
fig_width = num_columns * 6  # 5 inches per subplot column
fig_height = num_rows * 5    # 4 inches per subplot row

plt.figure(figsize=(fig_width, fig_height))

for i, column in enumerate(specific_variables):
    plt.subplot(num_rows, num_columns, i+1)
    plt.title(column, fontsize=25)
    sns.histplot(data_df[column], color="red", kde=True)  # Adding KDE for smooth distribution curve.
    plt.xlabel('')  # Remove x labels to prevent clutter
    plt.ylabel('')  # Remove y labels to prevent clutter

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# As observed in the histogram, noticeable spikes are apparent in the data. Excluding binary variables, the variables with the most prominent spikes are:
#
# - `age_first_funding_year`
# - `age_first_milestone_year`
# - `age_last_milestone_year`
# - `funding_total_usd`
# - `relationships`
#
# Lets now determinate how much our dataset is affected by unusual data points. To do that we will use the following function.

# #### Identifying Outliers: Function z-score

# **Z-scores** help us see how different a data point is from the average (mean) of the dataset. We will apply this method to one of the columns that has a higher number of outliers `age_first_milestone_year`.

# In[23]:


zscores=zscore(data_df["age_first_milestone_year"])
for threshold in range(1,6,1):
    print("Threshold value: {}". format(threshold))
    print("Number of outliers: {}".format(len(np.where(zscores>threshold)[0])))


# ##### Interpretation
#
# - Lower z-score thresholds mean we detect more outliers by looking at a broader range.
# - Changing the z-score threshold lets you adjust how sensitive the outlier detection is, depending on what you need.
#
# Analyzing the z-scores for the `age_first_milestone_year` column shows that changing the threshold affects how many outliers are detected.
#
# Now that we have identify outliers in our data, the next step is to normalize it. To do that we will use the **log-transformed** method.

# #### Handle Outliers: log-transformed method

# The log-transformed method involves taking the natural logarithm (ln) of each data point in a dataset. It's useful when data has a wide range of values, helping to balance and spread out the data.
#
# The following graph illustrates the dataset before and after applying log transformation.

# In[24]:


variables = ["age_first_funding_year", "age_last_funding_year", "age_first_milestone_year", "age_last_milestone_year"]

# Create a figure with a specific size
plt.figure(figsize=(17, 6), dpi=100)

# Loop through the list of variables
for i, variable in enumerate(variables):
    # Regular histogram
    plt.subplot(2, 4, i + 1)
    sns.histplot(data_df[variable], color="red", kde=True)
    plt.title(variable)

    # Log-transformed histogram
    plt.subplot(2, 4, i + 5)
    log_variable = np.log(data_df[variable] + 1)  # Adding 1 to avoid log(0)
    sns.histplot(log_variable, color="blue", kde=True)
    plt.title(f"log_{variable}")

plt.tight_layout()
plt.show()


# The distinction is evident; in the blue histogram, the peak is noticeably lower than in the red histogram.
#
#  When you use the logarithm on data, compresses the outliers closer to the rest of the values. This makes the distribution of the data look more like a balanced and symmetrical shape, similar to a bell curve.
#  - It's important to note that log transformation doesn't remove outliers entirely; it reduces their impact.
#
# Now that we've addressed negative values and outliers, let's shift our focus to transforming the other variables.

#
# Lets quickly inspect the first row of your DataFrame after sorting by index.

# In[25]:


data_df.sort_index().head(3)


# ### Creating Temporary Variables

# In this part, we'll demonstrate how useful it is to create temporary variables. The benefits of using temporary variables include ensuring the integrity of the original data, particularly during complex operations.
#
# Previously, we retained missing values in the `closed_at` variable to preserve information (as certain startups are still active without arbitrarily assigning a closure date). Creating temporary variables allows us to utilize the data without making modifications to it.

# In[26]:


data_df = data_df

#We convert the 'closed_at' column to datetime format, handling any conversion errors.
data_df['closed_at'] = pd.to_datetime(data_df['closed_at'], errors='coerce')

#Sort DataFrame by 'closed_at' in descending order revealing the most recent startup closures.
data_df = data_df.sort_values(by='closed_at', ascending=False )

#Find the last closing date
last_closed_date = data_df['closed_at'].dropna().iloc[0]

#Confirm is the right value
print("Last startup closing date:", last_closed_date)

#temporary variable, 'closed_temp,' is created to preserve non-null values of the 'closed_at' column for calculations.
closed_temp = data_df['closed_at'].copy()

#Fill the null values in 'closed_temp' with the last closed date(2013-10-30)
closed_temp.fillna(last_closed_date, inplace=True)

#Calculate the relative age based on 'founded_at' and 'closed_temp'
data_df['age'] = ((closed_temp - data_df['founded_at']).dt.days / 365.25).round(4)

#Missing values in 'closed_at' are replaced with 'x' to signify operating startups.
data_df['closed_at'] = data_df['closed_at'].fillna(value="x")

#'Missing values in 'closed_at' are replaced with 'x' to signify operating startups.
data_df['closed_at'] = data_df.closed_at.apply(lambda x: 1 if x =='x' else 0)


# ##### General Changes:
#
# - Create a temporary variable, `closed_temp` to hold non-null values from `closed_at`.
# - Fill the null values in `closed_temp` with the last closed date (2013-10-30).
# - Calculate the relative age based on `founded_at` and `closed_temp`.
# - Replace missing values in `closed_at` with 'x' to indicate operating startups.
#
# These operations serve the primary purpose of preparing and refining the dataset to facilitate subsequent analysis and reporting.

# In[27]:


data_df.sort_index().head(1)


# ### Transforming Non-numeric Data into Numeric Values

#
# Just like we did with our target variables previously, let's apply the **mapping** technique again to convert the remaining categorical features into numerical ones.

# #### Categorical to Numerical

# In[28]:


# List of categorical columns
categorical_columns = ['state_code', 'city', 'category_code','founded_year']
# Dictionary to store mappings
column_mappings = {}

# Create a function to generate mappings
def create_mapping(column):
    unique_values = data_df[column].unique()
    mapping = {value: i for i, value in enumerate(unique_values)}
    return mapping

# Apply mapping for each categorical column
for column in categorical_columns:
    mapping = create_mapping(column)
    data_df[column] = data_df[column].map(mapping)
    # Save mapping in the dictionary
    column_mappings[column] = mapping

# Now, column_mappings contains the desired dictionary structure
print(column_mappings)


# We proceed with converting the variables from dates to numerical values by transforming each specified column into days.

# #### Datatime to Numerical

# In[29]:


# Set a reference date. You could choose the earliest date in your dataset or a specific date.
reference_date = data_df[['founded_at', 'first_funding_at', 'last_funding_at']].min().min()

# List of columns to convert
columns_to_convert = ['founded_at', 'first_funding_at', 'last_funding_at']

# Convert each specified column into days since the reference date
for column in columns_to_convert:
    data_df[f'{column}_days'] = (data_df[column] - reference_date).dt.days

# Now, 'founded_at_days', 'first_funding_at_days', and 'last_funding_at_days' are numerical
    #columns representing the number of days from the reference date.
# 'closed_at' remains unchanged as it is already in numerical format (int64).

# Optionally, you might drop the original datetime columns if they're not needed anymore.
data_df.drop(['founded_at', 'first_funding_at', 'last_funding_at'], axis=1, inplace=True)


# #### Recap of Data Preprocessing ####

# After a comprehensive preprocessing transformation, we will outline the general steps we have taken with our data to improve its quality and gain more accurate insights.
#
# - Filtering Irrelevant Values: We simplified the dataframe by reducing the number of variables from 50 to 24, removing unnecessary ones.
#
# - Handling Missing Values: Identified and managed missing values through imputation techniques.
#
# - Addressing Negative Values: Utilized Box Plot and Scatter Plot methods to detect negative values, then handling them with the np.abs() function.
#
# - Managing Outliers: Employed Histogram and z-score methods for outlier detection and subsequently addressed them using log-transformed technique.
#
# - Temporary Variable Creation: Implemented temporary variables to facilitate complex calculations without altering the original dataset.
#
# - Conversion of Non-numeric Data: Converted non-numeric data into numeric values to improve analytical capabilities and model compatibility.
#
#
# After finally improving the quality of the data, we can now proceed to the most interesting part: getting some insights.

# ##  Ignore: (Changes for modeling)

# Additionally, by dropping the original datetime columns, the DataFrame becomes more concise and efficient for subsequent modeling tasks.

# In[30]:


data_df.sort_index().head(2)


# In[31]:


data_df = data_df.drop(['closed_at'], axis=1).copy()
data_df.head(2)


# In[32]:


data_df = data_df.drop(['age'], axis=1).copy()
data_df.head(2)


# In[33]:


numerical_features = data_df.select_dtypes(include=['number']).columns.tolist()
categorical_features = data_df.select_dtypes(include=['object']).columns.tolist()
datetime_features = data_df.select_dtypes(include=['datetime']).columns.tolist()

# Assuming the target variable is 'status'
target_variable = ['status']

# Print the lists along with the number of features
print("Numerical Features ({0}):".format(len(numerical_features)))
print(numerical_features)

print("\nCategorical Features ({0}):".format(len(categorical_features)))
print(categorical_features)

print("\nDatetime Features ({0}):".format(len(datetime_features)))
print(datetime_features)

print("\nTarget Variable ({0}):".format(len(target_variable)))
print(target_variable)


# In[34]:


data_df.to_csv('cleaned_data.csv', index=False)


# ##  Data Analysis

# Our objective is to understand the determinants impacting the likelihood of a startup being acquired.
#
# To achieve this, we will use a correlation **Heatmap** .
#
# Purpose of the visualization:
#
# - Provide a visual way to identify the variables that are highly correlated with the target `status`.
#
# We will compare two correlation methods, Pearson's and Spearman,selecting just the features with the highest absolute correlation values.

# #### Factors correlated with startup acquisitions

# In[35]:


def draw_heatmaps_side_by_side(data_df, target='status'):
    """
    Draws side-by-side heatmaps of the top 10 features correlated with a specified target column,
    using Spearman and Pearson correlation coefficients.

    :param data_df: DataFrame with the data
    :param target: String, the target column for correlation calculation
    """
    # Ensure target is in DataFrame
    if target not in data_df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns.")

    # Function to calculate and return top correlated features' correlation matrix
    def get_top_correlations(data_df, method):
        corr = data_df.corr(method=method)
        top_features = corr[target].abs().sort_values(ascending=False).head(11).index
        return corr.loc[top_features, top_features]

    # Calculate correlation matrices
    top_spearman_corr = get_top_correlations(data_df, 'spearman')
    top_pearson_corr = get_top_correlations(data_df, 'pearson')

    # Setup for dual heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))  # Adjust figsize as needed
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Diverging color map

    # Draw heatmaps
    for ax, corr, title in zip(axes, [top_spearman_corr, top_pearson_corr],
                               ['Spearman Correlation', 'Pearson Correlation']):
        sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f", annot_kws={"size": 12},
                    cbar_kws={"shrink": .5}, mask=np.triu(np.ones_like(corr, dtype=bool)))
        ax.set_title(f'Top 10 Features Correlated with {target} ({title})', fontsize=20)
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.show()

# Assuming 'data_df' is your DataFrame
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64']
numerical_df_1 = data_df.select_dtypes(include=numerics)

draw_heatmaps_side_by_side(numerical_df_1)


# From the heatmaps provided, which compare the Spearman and Pearson correlation coefficients for the top features correlated with the 'status' of startups, we can make our conclusion.
#
# - The features `relationships`, `milestones` achieved well as the number of `funding rounds` consistently show a positive correlation with `status` in both methods, implying that are potentially important indicators of startup success.

# Now that we have established their relationship, let's address each variable individually and proceed to explore the following key questions:

#  #### Does the number of milestones achieved impact a startup's chance of getting acquired?

# In[36]:


# Group your data by milestones and status and count the occurrences
grouped = data_df.groupby(["milestones", "status"]).size().reset_index(name="count")
grouped.columns = ["source", "target", "value"]
grouped['target'] = grouped['target'].map({0: 'closed', 1: 'acquired'})
grouped['source'] = grouped.source.map({0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 8: '8'})
links = pd.concat([grouped], axis=0)
unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
links['source'] = links['source'].map(mapping_dict)
links['target'] = links['target'].map(mapping_dict)

# Convert links DataFrame to dictionary
links_dict = links.to_dict(orient='list')

# Apply the color map to the links
# Directly use the 'acquired' and 'closed' labels to determine the color
link_colors = ['rgba(144, 238, 144, 0.5)' if target == 'acquired' else 'rgba(255, 0, 0, 0.5)' for target in grouped['target']]


fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=unique_source_target,
        color="white"
    ),
    link=dict(
        source=links_dict["source"],
        target=links_dict["target"],
        value=links_dict["value"],
        color=link_colors  # Apply the colors to the links
    )
)])

fig.update_layout(
    annotations=[
        dict(
            x=0.5,  # Set x to 0.5 for centering horizontally
            y=1.20,  # Adjust y as needed
            text="Analyzing the Correlation: Milestones Achievement vs Startup Status",
            showarrow=False,
            font=dict(
                size=18,
                family="Arial, sans-serif"  # Using a bold font family
            ),
            textangle=0,
            xref="paper",
            yref="paper"
        ),
        # Add a comma here after the closing brace of the first dictionary
        dict(
            x=-0.10,  # Position to the left of the diagram
            y=0.5,    # Vertically centered
            text="Milestones Achievement",  # Text for the left side
            showarrow=False,
            font=dict(size=15),
            textangle=-90,  # Rotate text to be vertical
            xref="paper",
            yref="paper"
        ),
        dict(
            x=1.05,  # Position to the right of the diagram
            y=0.5,   # Vertically centered
            text="Startup Status",  # Text for the right side
            showarrow=False,
            font=dict(size=15),
            textangle=-90,  # Rotate text to be vertical
            xref="paper",
            yref="paper"
        )
    ]
)

fig.show()


# The data indicates that startups achieving up to four milestones have a significantly higher likelihood of being acquired than those that dont.
#
# -  Specifically, there is a prominent increase in acquisitions for startups reaching two to four milestones, underscoring the importance of early milestone achievement in determining startup success.
#
# While the number of acquisitions decreases after reaching four milestones, the data consistently shows that achieving milestones positively influences a startup's likelihood of being acquired.
#
# This trend reinforces the value of setting and reaching milestones as a strategy for startups aiming to get acquired and succeed in the industry
#
# Having analyzed the significance of achieving milestones, let's now turn our attention to the types of investment.

# #### Do startups funded in rounds have a better chance of being acquired?

# In[37]:


# Filter the DataFrame for entries with a status of 1
d = data_df.loc[data_df['status'] == 1]

# Select only the relevant binary columns
f = d[["has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD"]]

# Melt the DataFrame to long format for use with countplot
melted_f = pd.melt(f)

# Map the binary values to strings
melted_f['value'] = melted_f['value'].map({1: 'acquired', 0: 'closed'})

# Create the countplot with horizontal bars
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
sns.countplot(data=melted_f, y='variable', hue='value', orient='h', palette="rainbow")

# Update the y-axis label with adjusted position
ax.set_ylabel("Investment Types", labelpad=20)  # Adjust labelpad for position
ax.set_xlabel("Startup Count", fontsize=10, labelpad=25)

# Create the title manually and adjust its position
ax.text(0.2, 1.05, "Investment Types for Acquired Startups", fontsize=15, transform=ax.transAxes, fontweight='bold')  # Adjust the x and y values

# Update the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Closed', 'Acquired'], title='Status')

plt.show()


# It shows that the majority of startups that received round D funding were acquired rather than closed.
#
# Investments from venture capital (VC) and rounds A, B, and C are more commonly associated with startups that were acquired, suggesting these types of investments might be linked to a higher likelihood of success.
#
# Now that we have confirm the importance of type of invesment is a factor for a startup to succed or not, let's analyse another variable that was highly correlated with our target.

#  #### What's the average funding for acquired vs. closed startups?

# In[38]:


# Define a list of pastel color codes
colors = ['#1f77b4', '#d62728']

# Filter out extreme outliers from the data
max_value = data_df['funding_total_usd'].quantile(0.99)  # Define a threshold (e.g., 99th percentile)
filtered_data = data_df[data_df['funding_total_usd'] < max_value]

# Create a histogram with the filtered data
fig = px.histogram(data_frame=filtered_data, x='funding_total_usd', color='status',
                   labels={'funding_total_usd': 'Total Funding (USD in Millions)'},
                   color_discrete_sequence=colors)

# Update layout for better visualization
fig.update_layout(
    title={
        'text': 'Distribution of Total Funding : Acquired vs Closed Startups',
        'y': 0.95,  # You can adjust this for vertical position
        'x': 0.5,  # You can adjust this for horizontal position
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {
            'size': 17,
            'color': 'black',  # You can change the color if you want
            'family': 'Arial, sans-serif', # You can change the font family if you want
        },
    },
    xaxis_title='Total Funding (USD in Millions)',
    yaxis_title='Frequency',
)
# Format the x-axis tick labels to display values in millions
fig.update_xaxes(tickformat=".2s", exponentformat="none")

# Show the figure
fig.show()


# From the histogram we can point the following:
#
# Most startups, whether acquired or closed, tend to have lower levels of total funding, with the likelihood of either acquisition or closure diminishing as funding amounts increase.
#
# This might imply that while sufficient funding is necessary for a startup's success, it is not the only determining factor, and there is a threshold beyond which more funding does not necessarily equate to greater chances of success.

# ##  Conclusion

# After a depth analysis of the variables, the factors that are related to startups being acquired are definitely the number of business relationships, number and timing of milestones reached, types of investment, alongside the historical record of funding, are significant factors to consider.
#
# Data preprocessing played a key role in turning raw information into practical insights. Critical steps as imputation techniques, identifying negative and missing values and carefully addressing outliers provided meaningful and accurate information about the underlying patterns.
#
# Keeping in mind that every technique will be used differently depending on the nature of the data.
#
# Visual representations enable us to decode intricate relationships, recognize trends, and identify outliers with ease, making it more accessible and intuitive for interpretation.
#
# By following this workflow, a data analysis project becomes more manageable, efficient, and produces reliable results that can inform critical decisions in various domains, including startup success prediction.

# ## Ignore: What is next

# Evaluating ML models and selecting the best-performing one is one of the main activities you do in pre-production.
#
# Hopefully, with this article, you’ve learned how to properly set up a model validation strategy and then how to choose a metric for your problem.
#
# You are ready to run a bunch of experiments and see what works.
#
# With that comes another problem of keeping track of experiment parameters, datasets used, configs, and results.
#
# And figuring out how to visualize and compare all of those models and results.
#
# For that, you may want to check out:
