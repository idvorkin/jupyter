# /// script
# [tool.marimo]
# version = "0.17.7"
#
# [tool.marimo.runtime]
# auto_instantiate = true
# on_cell_change = "autorun"
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Igor's Weight Analysis",
)


@app.cell
def _(mo):
    mo.md(r"""
    # Igor's Weight Tracking and Plotting Playground

    <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 8px; margin: 20px 0; font-size: 16px; font-weight: bold; text-align: center; border: 3px solid #cc0000;">
        ⚠️ CLICK THE RUN BUTTON (▶️) IN THE BOTTOM RIGHT CORNER TO START THE NOTEBOOK ⚠️
    </div>

    Back in the day, I wanted to learn matplotlib and visualization in Python. What better way to practice than with a fun, personal use case?
    This notebook started as a Jupyter playground for tracking my weight data and experimenting with different plotting techniques.

    Over time, it's evolved from basic matplotlib plots to include seaborn, Altair (Vega-Lite), and now runs in marimo instead of Jupyter.
    It's been a great way to learn data visualization while keeping tabs on my health journey!

    **Historical Note:** This notebook was originally created as [`Weight Analysis.ipynb`](https://github.com/idvorkin/jupyter/blob/b9080bc1980b5e13db6972b68bab145add0457f6/Weight%20Analysis.ipynb).
    That Jupyter version is now deprecated - please use this marimo version instead for better reactivity, cleaner diffs, and a more modern experience.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("ggplot")
    import seaborn as sns
    import matplotlib as mpl
    import arrow
    from matplotlib import animation
    from IPython.display import display
    from datetime import timedelta
    import altair as alt
    import marimo as mo

    # Suppress narwhals/altair compatibility warnings
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"You passed a `<class 'narwhals\.stable\.v1\.DataFrame'>` to `is_pandas_dataframe`.",
    )

    # '%matplotlib inline' command supported automatically in marimo
    return alt, animation, arrow, display, mo, mpl, np, pd, plt, sns, timedelta


@app.cell
def _():
    import json
    import os
    import sys

    def load_data_file(filename):
        """
        Load data file by trying multiple paths in order:
        1. Local relative path (data/filename)
        2. Local absolute paths (various common locations)
        3. GitHub raw URL

        This enables the notebook to work both locally and in WASM/browser environments.

        Args:
            filename: Name of the data file (e.g., "HealthAutoExport-2010-03-03-2025-05-30.json")

        Returns:
            Parsed JSON data
        """
        # Check if we're running in a WASM/Pyodide environment
        is_wasm = "pyodide" in sys.modules

        # Define paths to try in order
        local_paths = [
            f"data/{filename}",
            f"/home/developer/gits/jupyter/data/{filename}",
            f"./{filename}",
            filename,
        ]

        github_url = (
            f"https://raw.githubusercontent.com/idvorkin/jupyter/master/data/{filename}"
        )

        # Try local paths first (skip if in WASM)
        if not is_wasm:
            for path in local_paths:
                if os.path.exists(path):
                    print(f"✓ Loading data from local file: {path}")
                    with open(path, "r") as f:
                        return json.load(f)
                else:
                    print(f"✗ Not found: {path}")

        # Fall back to GitHub (works in both local and WASM)
        print(f"→ Loading data from GitHub: {github_url}")
        try:
            import urllib.request

            with urllib.request.urlopen(github_url) as response:
                data = json.loads(response.read().decode())
                print("✓ Successfully loaded from GitHub")
                return data
        except Exception as e:
            print(f"✗ Failed to load from GitHub: {e}")
            raise

    ## Getting and preparing the input file
    # Export data using HealthAutoExport
    # *********************************************************************************************************************
    # Using the comprehensive JSON export which has more recent and complete weight data
    # *********************************************************************************************************************
    data_filename = "HealthAutoExport-2010-03-03-2025-05-30.json"

    # Legacy CSV files (keeping for reference):
    # exported_and_trandformed_csv_file = "data/metrics-2024-09-01.csv"  # This file has weight data
    # exported_and_trandformed_csv_file = "data/metrics-2025-05-27.csv"  # This file has no weight data
    return (data_filename, load_data_file)


@app.cell
def _(arrow, data_filename, load_data_file, np, pd):
    # Load JSON data (tries local paths first, then GitHub)
    health_data = load_data_file(data_filename)

    # Extract weight data from JSON structure
    metrics = health_data["data"]["metrics"]
    weight_metric = None
    for health_metric in metrics:
        if health_metric["name"] == "weight_body_mass":
            weight_metric = health_metric
            break

    # Initialize variables
    idx_weight = "Weight"
    idx_date = "Date"
    _min_weight = 140
    idx_month_year = "month_year"
    idx_week_year = "week_year"
    idx_quarter_year = "quarter_year"

    if weight_metric is None:
        print("No weight data found in JSON!")
        # Create empty data structures
        df = pd.DataFrame()
        dfW = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        df_alltime = df
    else:
        # Convert weight data to DataFrame
        weight_records = []
        for record in weight_metric["data"]:
            weight_records.append({"Date": record["date"], "Weight": record["qty"]})

        df = pd.DataFrame(weight_records)
        print(f"Loaded {len(df)} weight records from JSON")

        # Parse dates - they have timezone info
        df[idx_date] = pd.to_datetime(df[idx_date], errors="coerce", utc=True)
        df = df.dropna(subset=[idx_date])
        print(f"After date parsing: {len(df)} rows")

        if len(df) > 0:
            print(f"Date range: {df[idx_date].min()} to {df[idx_date].max()}")
            print(
                f"Weight range: {df[idx_weight].min():.1f} to {df[idx_weight].max():.1f} lbs"
            )

            df = df.set_index(df[idx_date])
            df = df.sort_index()

            # Clean the data
            df = df.replace(0, np.nan)
            df = df[df.index > pd.to_datetime("2010-01-01", utc=True)]
            print(f"After 2010 filter: {len(df)} rows")

            df = df.dropna(subset=[idx_weight])
            print(f"After dropna weights: {len(df)} rows")

            df = df[df[idx_weight] > _min_weight]
            print(f"After weight > {_min_weight} filter: {len(df)} rows")

            if len(df) > 0:
                df[idx_month_year] = df.index.to_series().apply(
                    lambda t: arrow.get(t).format("MMM-YY") if pd.notna(t) else None
                )
                df[idx_week_year] = df.index.to_series().apply(
                    lambda t: f"{t.week}-{t.year - 2000}" if pd.notna(t) else None
                )
                df[idx_quarter_year] = df.index.to_series().apply(
                    lambda t: arrow.get(t).ceil("quarter").format("MMM-YY")
                    if pd.notna(t)
                    else None
                )

                dfW = df[idx_weight].copy()
                dfW.index = df.index
                df_alltime = df
            else:
                dfW = pd.Series(
                    dtype="float64", name=idx_weight, index=pd.DatetimeIndex([])
                )
                df_alltime = df
        else:
            dfW = pd.Series(
                dtype="float64", name=idx_weight, index=pd.DatetimeIndex([])
            )
            df_alltime = df
    return (
        df,
        dfW,
        df_alltime,
        idx_date,
        idx_month_year,
        idx_quarter_year,
        idx_week_year,
        idx_weight,
    )


@app.cell
def _(alt, display, idx_weight, mpl, plt, sns):
    def box_plot_weight_mpl(df, x, title=""):
        # In theory can use plot.ly (not free)  or Bokeh (not mpl compatible) but issues. So setting dimensions old school.
        # Manually setting the weight and width - using larger sizes for full width display
        height_in_inches = 10
        mpl.rc("figure", figsize=(20, height_in_inches))

        # Create a custom color palette
        palette = sns.color_palette("husl", len(df[x].unique()))
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Create the boxplot with the custom palette
        ax = sns.boxplot(
            x=x, y=idx_weight, data=df, palette=palette, hue=x, legend=False
        )

        # Set title and labels with improved styling
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Weight (lbs)", fontsize=12)

        # Improve the overall style
        sns.set_style("whitegrid")
        plt.tight_layout()

        # Show the plot
        plt.show()

    def box_plot_weight_vegas(df, x, title, domain=(150, 250)):
        height_in_inches = 600  # Larger height for better visibility
        y_min = min(df[idx_weight]) - 10
        y_max = max(df[idx_weight]) + 10
        domain = (y_min, y_max)
        display(domain)
        c = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                y=alt.Y(idx_weight, scale=alt.Scale(domain=domain, clamp=True)),
                x=alt.X(x, type="temporal", sort="ascending"),
            )
            .properties(width="container", height=height_in_inches, title=title)
            .interactive()
        )
        display(c)

    return (box_plot_weight_mpl,)


@app.cell
def _(
    box_plot_weight_mpl,
    df,
    df_alltime,
    display,
    idx_month_year,
    idx_quarter_year,
    idx_week_year,
    pd,
):
    _earliest = df.index[-1] - pd.DateOffset(years=2)  # Use index instead of .Date
    display(_earliest)
    box_plot_weight = box_plot_weight_mpl
    box_plot_weight(df[_earliest:], idx_month_year, title="Recent weight by month")
    box_plot_weight(df[_earliest:], idx_week_year, title="Recent weight by week")
    box_plot_weight(df_alltime, idx_month_year, "Weight by Month")
    box_plot_weight(df_alltime, idx_quarter_year, "Weight by Quarter")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time Series Analysis using resampling

    **Note:** The red dashed line on the charts below marks **February 2024**, when I started taking tirzepatide (Mounjaro/Zepbound).
    This GLP-1 medication has been a game-changer for weight management.
    You can read more about my experience at [idvork.in/terzepatide](https://idvork.in/terzepatide).
    """)
    return


@app.cell
def _(alt, df, display, idx_date, idx_weight, pd):
    print("Scroll to see year markers, select in index to zoom in")
    metric = idx_weight

    def make_domain(df):
        y_min = min(df[idx_weight])
        y_max = max(df[idx_weight])
        distance = y_max - y_min
        buffer = 0.05 * distance
        buffer = min(buffer, 5)
        domain = (y_min - buffer, y_max + buffer)
        return domain

    def graph_weight_as_line(df, freq, domain):
        domain = make_domain(df)
        # Map frequency names to pandas frequency codes
        freq_map = {"Month": "ME", "Week": "W"}
        pd_freq_value = freq_map.get(freq, freq[0])
        df_group_time = df.copy()[metric].resample(pd_freq_value)
        t1 = df_group_time.count().reset_index()
        df_to_graph = t1.drop(columns=metric)
        for q in [0.25, 0.5, 0.9]:
            df_to_graph[f"p{q * 100}"] = df_group_time.quantile(q).reset_index()[metric]
        df_melted = df_to_graph.melt(id_vars=[idx_date])
        chart_height = 500  # Larger height for better visibility
        selection = alt.selection_point(fields=["variable"], bind="legend")

        # Create the main line chart
        line_chart = (
            alt.Chart(df_melted)
            .mark_line(point=True)
            .encode(
                y=alt.Y("value", title="", scale=alt.Scale(domain=domain)),
                x=alt.X(f"{idx_date}:T"),
                color=alt.Color("variable"),
                tooltip=[alt.Tooltip(f"{idx_date}:T"), alt.Tooltip("value:Q")],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            )
            .add_params(selection)
        )

        # Add vertical line for tirzepatide start date (February 2024)
        tirzepatide_date = pd.Timestamp("2024-02-01", tz="UTC")
        rule = (
            alt.Chart(pd.DataFrame({idx_date: [tirzepatide_date]}))
            .mark_rule(color="red", strokeDash=[5, 5], size=2)
            .encode(x=f"{idx_date}:T")
        )

        # Add text annotation
        text = (
            alt.Chart(
                pd.DataFrame(
                    {idx_date: [tirzepatide_date], "label": ["Started Tirzepatide"]}
                )
            )
            .mark_text(align="left", dx=5, dy=-10, color="red", fontSize=12)
            .encode(x=f"{idx_date}:T", text="label")
        )

        # Combine all layers
        c = (
            (line_chart + rule + text)
            .properties(
                width="container",
                height=chart_height,
                title=f"{metric} By {freq}",
            )
            .interactive()
        )
        display(c)
        return c

    _earliest = df.index[-1] - pd.DateOffset(years=1)  # Use index instead of .Date
    graph_weight_as_line(df[_earliest:], "Week", (180, 205))
    return (graph_weight_as_line,)


@app.cell
def _(df, graph_weight_as_line):
    graph_weight_as_line(df, "Month", (150, 240))
    return


@app.cell
def _(df, graph_weight_as_line):
    graph_weight_as_line(df, "Week", (150, 240))
    return


@app.cell
def _(dfW):
    dfM = dfW.resample("W").median()
    return (dfM,)


@app.cell
def _(animation, dfM, mo, plt, timedelta):
    import datetime

    anim_year_base = 2015
    years_to_plot = datetime.datetime.now().year - anim_year_base + 1
    anim_fig_size = (16, 7)
    fig = plt.figure(figsize=anim_fig_size)
    ax = fig.add_subplot(1, 1, 1)
    dfRelevant = dfM[f"{anim_year_base}" : f"{anim_year_base + years_to_plot}"]

    _min_weight = dfRelevant.min() - 5
    max_weight = dfRelevant.max() + 5
    dfM[f"{anim_year_base}" : f"{anim_year_base}"].plot(
        title="Title Over Written",
        figsize=anim_fig_size,
        ylim=(_min_weight, max_weight),
        ax=ax,
    )
    ax.set_ylabel("lbs")
    ax.set_xlabel("")

    def animate(i):
        year = f"{anim_year_base + i}"
        return (
            dfM[f"{anim_year_base}" : year].plot(title=f"Weight through {year}").lines
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=years_to_plot,
        interval=timedelta(seconds=2).seconds * 1000,
        blit=False,
    )
    # Use to_jshtml() instead of to_html5_video() - works in browser without ffmpeg
    mo.Html(anim.to_jshtml())
    return


@app.cell
def _(dfM):
    dfM.min()
    return


if __name__ == "__main__":
    app.run()
