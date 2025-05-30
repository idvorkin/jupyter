import marimo

__generated_with = "0.13.15"
app = marimo.App()


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
    from IPython.display import HTML, display
    from datetime import timedelta
    import altair as alt
    import marimo as mo

    # '%matplotlib inline' command supported automatically in marimo
    return (
        HTML,
        alt,
        animation,
        arrow,
        display,
        mo,
        mpl,
        np,
        pd,
        plt,
        sns,
        timedelta,
    )


@app.cell
def _():
    ## Getting and preparing the input file
    # Export data using HealthAutoExport
    # *********************************************************************************************************************
    # NOTE: There is a trailing comma on the csv on the export file --> SO: You need to add a junk column in the header
    # *********************************************************************************************************************
    # exported_and_trandformed_csv_file = "data/metrics-2024-03-08.csv"
    # exported_and_trandformed_csv_file = "data/metrics-2024-09-01.csv"
    exported_and_trandformed_csv_file = "data/metrics-2025-05-27.csv"
    return (exported_and_trandformed_csv_file,)


@app.cell
def _(arrow, exported_and_trandformed_csv_file, np, pd):
    df = pd.read_csv(exported_and_trandformed_csv_file, sep=",")
    idx_weight, _min_weight = ("Weight/Body Mass (lb)", 140)
    idx_date = "Date"
    df[idx_date] = pd.to_datetime(df[idx_date])
    df = df.set_index(df[idx_date])
    df = df.sort_index()
    idx_month_year = "month_year"
    df[idx_month_year] = df.index.to_series().apply(
        lambda t: arrow.get(t).format("MMM-YY")
    )
    idx_week_year = "week_year"
    df[idx_week_year] = df.index.to_series().apply(
        lambda t: f"{t.week}-{t.year - 2000}"
    )
    idx_quarter_year = "quarter_year"
    df[idx_quarter_year] = df.index.to_series().apply(
        lambda t: arrow.get(t).ceil("quarter").format("MMM-YY")
    )
    df = df.replace(0, np.nan)
    df = df[df.index > "2010-01-01"]
    df = df[df[idx_weight] > _min_weight]
    dfW = df[idx_weight]
    df_alltime = df
    # display(df)
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
def _(df):
    df
    return


@app.cell
def _(alt, display, idx_weight, mpl, plt, sns):
    def box_plot_weight_mpl(df, x, title=""):
        # In theory can use plot.ly (not free)  or Bokeh (not mpl compatible) but issues. So setting dimensions old school.
        # Manually setting the weight and width.
        height_in_inches = 8
        mpl.rc("figure", figsize=(4 * height_in_inches, height_in_inches))

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
        height_in_inches = 4 * 60  # todo figure out how to get this by calculation
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
            .properties(
                width=4 * height_in_inches, height=height_in_inches, title=title
            )
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
    _earliest = df.iloc[-1].Date - pd.DateOffset(years=2)
    display(_earliest)
    box_plot_weight = box_plot_weight_mpl
    box_plot_weight(df[_earliest:], idx_month_year, title="Recent weight by month")
    box_plot_weight(df[_earliest:], idx_week_year, title="Recent weight by week")
    box_plot_weight(df_alltime, idx_month_year, "Weight by Month")
    box_plot_weight(df_alltime, idx_quarter_year, "Weight by Quarter")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Time Series Analysis using resampling""")
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
        pd_freq_value = freq[0]
        df_group_time = df.copy()[metric].resample(pd_freq_value)
        t1 = df_group_time.count().reset_index()
        df_to_graph = t1.drop(columns=metric)
        for q in [0.25, 0.5, 0.9]:
            df_to_graph[f"p{q * 100}"] = df_group_time.quantile(q).reset_index()[metric]
        df_melted = df_to_graph.melt(id_vars=[idx_date])
        height_in_inches = 60
        selection = alt.selection_point(fields=["variable"], bind="legend")
        c = (
            alt.Chart(df_melted)
            .mark_line(point=True)
            .encode(
                y=alt.Y("value", title="", scale=alt.Scale(domain=domain)),
                x=alt.X(f"{idx_date}:T"),
                color=alt.Color("variable"),
                tooltip=[alt.Tooltip(f"{idx_date}:T"), alt.Tooltip("value:Q")],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            )
            .properties(
                width=16 * height_in_inches,
                height=6 * height_in_inches,
                title=f"{metric} By {freq}",
            )
            .interactive()
            .add_params(selection)
        )
        display(c)
        return c

    _earliest = df.iloc[-1].Date - pd.DateOffset(years=1)
    graph_weight_as_line(df[_earliest:], "Week", (180, 205))
    for freq in "Month Week".split():
        graph_weight_as_line(df, freq, (150, 240))
    return


@app.cell
def _(dfW):
    dfM = dfW.resample("W").median()
    return (dfM,)


@app.cell
def _(HTML, animation, dfM, plt, timedelta):
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
    HTML(anim.to_html5_video())
    return


@app.cell
def _(dfM):
    dfM.min()
    return


@app.cell
def _():
    from pandasai import PandasAI
    from pandasai.llm.openai import OpenAI

    return OpenAI, PandasAI


@app.cell
def _(OpenAI, PandasAI, pd):
    def setup_gpt():
        import os
        import json

        PASSWORD = "replaced_from_secret_box"
        with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
            SECRETS = json.load(json_data)
            PASSWORD = SECRETS["openai"]
        return PASSWORD

    model = "gpt-3.5-turbo"
    llm = OpenAI(setup_gpt(), model=model)
    pandas_ai = PandasAI(llm, verbose=True)
    exported_and_trandformed_csv_file_1 = "data/weight.csv"
    df_1 = pd.read_csv(exported_and_trandformed_csv_file_1)
    pandas_ai.run(df_1, "Remove rows with weights less then 100 from", show_code=True)
    pandas_ai.run(df_1, prompt="What is the average weight?", show_code=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
