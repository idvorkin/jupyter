import marimo as mo
# import micropip # Commented out for direct python execution

# # Install necessary packages
# await micropip.install([
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "arrow",
#     "altair",
#     "icecream",
#     "pandasai",
#     "openai"
# ])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np # Redundant, but kept for consistency with notebook
import matplotlib as mpl
import arrow
from matplotlib import animation, rc
# from IPython.display import HTML # Will be replaced by mo.Html for animation
from datetime import timedelta
import altair as alt
from icecream import ic
# from pandasai import PandasAI # Commented out for now
# from pandasai.llm.openai import OpenAI # Commented out for now

matplotlib.style.use("ggplot")

print("--- Starting Marimo Script Simulation ---")

mo.md("## Data Loading and Initial Processing")
print("\n--- Section: Data Loading and Initial Processing ---")

# Data Loading Cell 1
mo.md("## Data Loading and Initial Definitions")
print("\n--- Cell: Data Loading and Initial Definitions ---")
# Getting and preparing the input file
exported_and_trandformed_csv_file_url = "https://raw.githubusercontent.com/idvorkin/jupyter/master/data/metrics-2024-09-01.csv"
df_loaded = pd.read_csv(exported_and_trandformed_csv_file_url, sep=",")
idx_weight, min_weight = "Weight/Body Mass (lb)", 140
idx_date = "Date"
df_loaded[idx_date] = pd.to_datetime(df_loaded[idx_date])
df_processed_part1 = df_loaded # Intermediate step
print("--- df_processed_part1 (after initial load and date conversion) ---")
print(df_processed_part1.head())
df_processed_part1.info()


# Data Loading Cell 2: Date/Index Processing and Time Period Groups
mo.md("### Date Processing and Time Period Groups")
print("\n--- Cell: Date Processing and Time Period Groups ---")
df_processed_part2 = df_processed_part1.set_index(df_processed_part1[idx_date])
df_processed_part2 = df_processed_part2.sort_index()

idx_month_year = "month_year"
df_processed_part2[idx_month_year] = df_processed_part2.index.to_series().apply(lambda t: arrow.get(t).format("MMM-YY"))

idx_week_year = "week_year"
df_processed_part2[idx_week_year] = df_processed_part2.index.to_series().apply(lambda t: f"{t.isocalendar().week}-{t.year-2000}")

idx_quarter_year = "quarter_year"
df_processed_part2[idx_quarter_year] = df_processed_part2.index.to_series().apply(
    lambda t: arrow.get(t).ceil("quarter").format("MMM-YY")
)
print("\n--- df_processed_part2 (after adding time period groups) ---")
print(df_processed_part2.head())

# Data Loading Cell 3: Data Cleaning
mo.md("### Data Cleaning")
print("\n--- Cell: Data Cleaning ---")
df_processed_part3 = df_processed_part2.replace(0, np.nan)
df_processed_part3 = df_processed_part3[df_processed_part3.index > "2010-01-01"]
df_processed_part3 = df_processed_part3[df_processed_part3[idx_weight] > min_weight]
df_processed = df_processed_part3 # Final df_processed
print("\n--- df_processed (after cleaning) ---")
print(df_processed.head())

# Data Loading Cell 4: DataFrame Aliases and Resampling
mo.md("### Derived DataFrames for Analysis")
print("\n--- Cell: Derived DataFrames for Analysis ---")
dfW = df_processed[idx_weight]
df_alltime = df_processed.copy()
dfM = dfW.resample("W").median()
# Display a small piece of derived data to confirm cell execution
print(f"\ndfM (median weekly weight) has {len(dfM)} entries. Min: {dfM.min()}, Max: {dfM.max()}")
mo.md(f"dfM (median weekly weight) has {len(dfM)} entries. Min: {dfM.min()}, Max: {dfM.max()}")

# Data Loading Cell 5: Display df_alltime head
mo.md("### Processed DataFrame Head (df_alltime)")
print("\n--- Cell: Processed DataFrame Head (df_alltime) ---")
print("\ndf_alltime.head():")
print(df_alltime.head())
print("\ndf_alltime.info():")
df_alltime.info()
print("\ndf_alltime.describe():")
print(df_alltime.describe())
# mo.ui.table(df_alltime.head()) # Marimo UI, will not render in bash


mo.md("## Plotting Function Definitions")
print("\n--- Section: Plotting Function Definitions ---")

mo.md("### Matplotlib Box Plot Function")
print("\n--- Cell: Matplotlib Box Plot Function Definition ---")
def box_plot_weight_mpl(df_local, x_col, title=""): 
    fig, ax = plt.subplots(figsize=(4 * 8, 8)) 
    palette = sns.color_palette("husl", len(df_local[x_col].unique()))
    # plt.xticks(rotation=45, ha='right') # This affects global state, better on ax
    ax.tick_params(axis='x', rotation=45) # Set rotation on the axes object
    sns.boxplot(x=x_col, y=idx_weight, data=df_local, palette=palette, hue=x_col, legend=False, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Weight (lbs)", fontsize=12)
    sns.set_style("whitegrid")
    plt.tight_layout()
    plt.close(fig) # Close plot to prevent display in bash
    return fig 

mo.md("### Altair Box Plot Function")
print("\n--- Cell: Altair Box Plot Function Definition ---")
def box_plot_weight_vegas(df_local, x_col, title, domain=(150, 250)): 
    height_in_inches = 4 * 60
    alt_domain = list(domain)
    c = (
        alt.Chart(df_local)
        .mark_boxplot()
        .encode(
            y=alt.Y(idx_weight, scale=alt.Scale(domain=alt_domain, clamp=True)), 
            x=alt.X(x_col, type='temporal', sort='ascending')
        )
        .properties(width=4 * height_in_inches, height=height_in_inches, title=title)
        .interactive()
    )
    return c

mo.md("### Altair Time Series Plot Functions")
print("\n--- Cell: Altair Time Series Plot Function Definitions ---")
def make_domain_altair(df_local_altair): 
    y_min = min(df_local_altair[idx_weight]) 
    y_max = max(df_local_altair[idx_weight])
    distance = (y_max - y_min)
    buffer = 0.05*distance
    buffer = min(buffer,5) 
    domain_altair=(y_min-buffer,y_max+buffer)
    return domain_altair

def graph_weight_as_line(df_local_chart, freq, title_suffix): 
    domain_chart = make_domain_altair(df_local_chart)
    
    if freq == "Week":
        pd_freq_value = "W"
    elif freq == "Month":
        pd_freq_value = "ME" 
    else:
        pd_freq_value = freq

    df_group_time = df_local_chart.copy()[[idx_weight]].resample(pd_freq_value)
    t1 = df_group_time.count().reset_index()
    df_to_graph = t1.drop(columns=idx_weight)
    for q_val in [0.25, 0.5, 0.9]: 
        df_to_graph[f"p{q_val*100}"] = df_group_time.quantile(q_val).reset_index()[idx_weight]

    df_melted = df_to_graph.melt(id_vars=[idx_date])

    height_in_inches = 60
    selection = alt.selection_point(fields=["variable"], bind="legend")

    chart_title = f"{idx_weight} By {freq}{title_suffix}"

    c_chart = ( 
        alt.Chart(df_melted)
        .mark_line(point=True)
        .encode(
            y=alt.Y("value", title="", scale=alt.Scale(domain=list(domain_chart))), 
            x=alt.X(f"{idx_date}:T"),
            color=alt.Color("variable"),
            tooltip=[alt.Tooltip(f"{idx_date}:T"), alt.Tooltip("value:Q")],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .properties(
            width=16 * height_in_inches,
            height=6 * height_in_inches,
            title=chart_title,
        )
        .interactive()
    ).add_params(selection)
    return c_chart

mo.md("### Matplotlib Animation Functions")
print("\n--- Cell: Matplotlib Animation Function Definitions ---")
# Global for animation - this is okay in Marimo as cells define dependencies
anim_fig_size = (16, 7) # Moved here as it's specific to this animation setup
fig_anim, ax_anim = plt.subplots(figsize=anim_fig_size)
line, = ax_anim.plot([], [], lw=2) # Line object for animation update

def init_anim():
    line.set_data([], [])
    return (line,)

def animate_mpl(i, fig, ax, dfM_local, min_w, max_w, base_year_anim): 
    year_anim = f"{base_year_anim+i}"
    current_data = dfM_local[f"{base_year_anim}":year_anim]
    
    ax.clear() # Clear and redraw for animation
    current_data.plot(
        title=f"Weight through {year_anim}",
        ylim=(min_w, max_w),
        ax=ax
    )
    ax.set_ylabel("lbs")
    ax.set_xlabel("")
    return ax.lines # Return changed artists

mo.md("### PandasAI Setup Function")
print("\n--- Cell: PandasAI Setup Function Definition ---")
def setup_gpt_marimo(): 
    return mo.ui.text(label="OpenAI API Key", kind="password")


mo.md("## Matplotlib Box Plots")
print("\n--- Section: Matplotlib Box Plots ---")
mo.md("These plots show the distribution of weight over different time periods using Matplotlib.")

# Matplotlib Box Plot 1
print("\n--- Cell: Matplotlib Box Plot 1 ---")
earliest_mpl = df_alltime.iloc[-1].Date - pd.DateOffset(years=2) 
plot1_fig = box_plot_weight_mpl(df_alltime[earliest_mpl:], idx_month_year, title="Recent weight by month")
print("plot1_fig created:", type(plot1_fig))
# plot1_fig # In Marimo, this would display the plot

# Matplotlib Box Plot 2
print("\n--- Cell: Matplotlib Box Plot 2 ---")
plot2_fig = box_plot_weight_mpl(df_alltime[earliest_mpl:], idx_week_year, title="Recent weight by week")
print("plot2_fig created:", type(plot2_fig))
# plot2_fig

# Matplotlib Box Plot 3
print("\n--- Cell: Matplotlib Box Plot 3 ---")
plot3_fig = box_plot_weight_mpl(df_alltime, idx_month_year, "Weight by Month")
print("plot3_fig created:", type(plot3_fig))
# plot3_fig

# Matplotlib Box Plot 4
print("\n--- Cell: Matplotlib Box Plot 4 ---")
plot4_fig = box_plot_weight_mpl(df_alltime, idx_quarter_year, "Weight by Quarter")
print("plot4_fig created:", type(plot4_fig))
# plot4_fig

# Overall title for the next section
mo.md("# Time Series Analysis using resampling")
print("\n--- Section: Time Series Analysis using resampling ---")

# Altair Time Series Plots
mo.md("The following plots show weight trends (P25, P50, P90 quantiles) over time, resampled weekly and monthly. Scroll to see year markers, select in index to zoom in.")
print("\n--- Cell: Altair Time Series Plots ---")
earliest_ts_altair = df_alltime.iloc[-1].Date - pd.DateOffset(years=1) 

chart1_altair = graph_weight_as_line(df_alltime[earliest_ts_altair:], "Week", " (Last Year)")
print("chart1_altair created:", type(chart1_altair))
# chart1_altair 

chart2_altair = graph_weight_as_line(df_alltime, "Month", " (All Time)")
print("chart2_altair created:", type(chart2_altair))
# chart2_altair 

chart3_altair = graph_weight_as_line(df_alltime, "Week", " (All Time)")
print("chart3_altair created:", type(chart3_altair))
# chart3_altair 


mo.md("## Matplotlib Animation")
print("\n--- Section: Matplotlib Animation ---")
mo.md("This animation shows the progression of median weekly weight year by year.")
# Matplotlib Animation Setup
print("\n--- Cell: Matplotlib Animation Setup ---")
anim_year_base = 2015
years_to_plot = 2024 - anim_year_base + 1 

dfRelevant_anim = dfM[f"{anim_year_base}":f"{anim_year_base+years_to_plot-1}"] 
min_weight_anim = dfRelevant_anim.min() - 5
max_weight_anim = dfRelevant_anim.max() + 5

ax_anim.set_xlim([dfRelevant_anim.index.min(), dfRelevant_anim.index.max()])
ax_anim.set_ylim([min_weight_anim, max_weight_anim])
ax_anim.set_title("Weight Animation") # Initial title, will be updated by animate_mpl
ax_anim.set_ylabel("lbs")

# Matplotlib Animation Execution
print("\n--- Cell: Matplotlib Animation Execution ---")
anim = animation.FuncAnimation(
    fig_anim,
    animate_mpl,
    init_func=init_anim, 
    frames=years_to_plot,
    fargs=(fig_anim, ax_anim, dfM, min_weight_anim, max_weight_anim, anim_year_base),
    interval=timedelta(seconds=2).seconds * 1000, # Corrected here
    blit=False, 
)
html_video = anim.to_html5_video()
print("Matplotlib animation created and to_html5_video() executed.")
plt.close(fig_anim) # Close animation figure
# mo.Html(html_video)

# dfM.min() Display
mo.md("## Minimum Median Weekly Weight (dfM.min())")
print("\n--- Cell: Minimum Median Weekly Weight (dfM.min()) ---")
print("dfM.min():", dfM.min())
# dfM.min()

# PandasAI Setup
mo.md("## PandasAI Setup (Requires API Key)")
print("\n--- Section: PandasAI Setup ---")
mo.md("Enter your OpenAI API key below to enable natural language data analysis with PandasAI. The actual analysis calls are commented out and can be enabled if an API key is provided.")

print("\n--- Cell: PandasAI API Key Input ---")
api_key_input = setup_gpt_marimo()
print("api_key_input (type):", type(api_key_input))
# api_key_input 

# Conditional PandasAI initialization and usage (commented out for now)
# This cell would be run by the user after providing the API key.
# mo.md("### PandasAI Analysis (Example - Uncomment to run)")
# print("\n--- Cell: PandasAI Analysis (Conditional) ---")
# pai_output = None
# # if api_key_input.value and api_key_input.value != "":
# #     try:
# #         llm = OpenAI(api_key_input.value, model="gpt-3.5-turbo")
# #         pandas_ai = PandasAI(llm, verbose=True, enable_cache=False) # Disabled cache for demo
# #         df_pai_example = df_alltime.copy()
# #         # Example Query (adjust as needed)
# #         # pai_output = pandas_ai.run(df_pai_example, "What is the average weight between January 2022 and December 2023?", show_code=True)
# #         # pai_output = pandas_ai.run(df_pai_example, "Remove rows with weights less then 100 from", show_code=True)
# #         pai_output = "PandasAI initialized. Uncomment a query to run."
# #         print("pai_output (if block):", pai_output)
# #     except Exception as e:
# #         pai_output = f"Error initializing PandasAI or running query: {e}"
# #         print("pai_output (exception block):", pai_output)
# # else:
# #     pai_output = "PandasAI not initialized. Please enter API key above."
# #     print("pai_output (else block):", pai_output) 
# # # pai_output

print("\n--- Script execution simulation complete (PandasAI logic and micropip commented out). ---")

# Commented out PandasAI run calls from original notebook
# prompt_pai = f"""
# Graph  P5, P50 and P90 Weight over time, let the x axis be every quarter
# """
# # pandas_ai.run(df_alltime, prompt=prompt_pai, show_code=True) # Use df_alltime or a copy
