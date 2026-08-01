from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import src.forecasting as ts


DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="Time Series Forecasting Toolkit",
    page_icon="📈",
    layout="wide",
)

st.title("Time Series Forecasting Toolkit")

st.markdown(
    """
    <style>
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p {
        font-size: 20px !important;
        font-weight: bold !important;
        font-family: inherit !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        font-size: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
### Interactive forecasting and backtesting

Compare time-series forecasting methods using repository data, an uploaded file,
a URL, or live currency data. Forecast future values or reserve the most recent
observations for backtesting against unseen data.
"""
)


# ---------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------

input_method = st.radio(
    "Choose data source",
    [
        "Repository data",
        "Upload from computer",
        "Import from URL",
        "Currency rates",
    ],
    horizontal=True,
)

dataframe = None
source_label = ""

if input_method == "Repository data":

    data_files = ts.available_data_files(DATA_DIR)

    if not data_files:
        st.error(
            f"No supported files were found in `{DATA_DIR}`. "
            "Add CSV, DAT, TXT or TSV files to the data directory."
        )
        st.stop()

    selected_filename = st.selectbox(
        "Select data file from project repository",
        [path.name for path in data_files],
    )

    try:
        dataframe = ts.read_data_file(
            DATA_DIR / selected_filename
        )
        source_label = selected_filename
    except Exception as error:
        st.error(
            f"Could not read `{selected_filename}`: {error}"
        )
        st.stop()

elif input_method == "Upload from computer":

    uploaded_file = st.file_uploader(
        "Choose a CSV or text data file",
        type=["csv", "txt", "dat", "tsv"],
    )

    if uploaded_file is None:
        st.info("Awaiting file upload.")
        st.stop()

    try:
        dataframe = ts.read_data_file(uploaded_file)
        source_label = uploaded_file.name
    except Exception as error:
        st.error(
            f"Could not read uploaded file: {error}"
        )
        st.stop()

elif input_method == "Import from URL":

    url_input = st.text_input(
        "Dataset web URL",
        placeholder="https://example.com/data.csv",
    )

    if not url_input:
        st.info("Please enter a dataset URL.")
        st.stop()

    try:
        dataframe = ts.read_data_file(url_input)
        source_label = url_input
    except Exception as error:
        st.error(
            f"Could not read dataset from URL: {error}"
        )
        st.stop()

else:

    currencies = [
        "USD",
        "EUR",
        "GBP",
        "NZD",
        "AUD",
        "JPY",
        "CAD",
        "CHF",
    ]

    currency_columns = st.columns([1, 1, 1.3])

    with currency_columns[0]:
        currency_from = st.selectbox(
            "From",
            currencies,
            index=2,
        )

    with currency_columns[1]:
        currency_to = st.selectbox(
            "To",
            currencies,
            index=3,
        )

    with currency_columns[2]:
        days_back = st.number_input(
            "Days back",
            min_value=7,
            max_value=5000,
            value=365,
            step=1,
        )

    if currency_from == currency_to:
        st.error(
            "Choose two different currencies."
        )
        st.stop()

    try:
        dataframe = ts.download_currency_data(
            currency_from=currency_from,
            currency_to=currency_to,
            days_back=int(days_back),
        )
        source_label = (
            f"{currency_from}/{currency_to} exchange rate"
        )
    except Exception as error:
        st.error(
            f"Could not download currency data: {error}"
        )
        st.stop()


# ---------------------------------------------------------------------
# Data inspection and column selection
# ---------------------------------------------------------------------

if dataframe is None or dataframe.empty:
    st.error("The selected data source returned no rows.")
    st.stop()

with st.expander(
    "Preview data",
    expanded=False,
):
    st.dataframe(
        dataframe,
        use_container_width=True,
    )

columns = list(dataframe.columns)

if len(columns) < 2:
    st.error(
        "The dataset must contain at least two columns."
    )
    st.stop()

column_row = st.columns([1, 1, 1])

with column_row[0]:
    date_column = st.selectbox(
        "Select date column",
        columns,
        index=0,
    )

numeric_columns = [
    column
    for column in columns
    if column != date_column
    and pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).notna().sum() >= 3
]

if not numeric_columns:
    st.error(
        "The dataset must contain at least one numeric value column."
    )
    st.stop()

with column_row[1]:
    value_column = st.selectbox(
        "Select value column",
        numeric_columns,
        index=0,
    )

with column_row[2]:
    replace_zeroes = st.selectbox(
        "Replace zeroes with missing values?",
        ["No", "Yes"],
        index=0,
    ) == "Yes"


# ---------------------------------------------------------------------
# Optional filtering
# ---------------------------------------------------------------------


st.markdown(
    "<p style='font-size: 20px; font-weight: bold; "
    "margin-bottom: 5px;'>Optional filtering</p>",
    unsafe_allow_html=True,
)

st.caption(
    "Use this when the dataset contains multiple time series in one file. "
    "Choose a field and value to analyse only that subset."
)



filter_columns = [
    column
    for column in columns
    if column not in [date_column, value_column]
]

apply_filter = False

if filter_columns:

    filter_row = st.columns([0.7, 1.2, 1.4])

    with filter_row[0]:
        apply_filter = st.selectbox(
            "Filter data?",
            ["No", "Yes"],
            index=0,
        ) == "Yes"

    with filter_row[1]:
        filter_column = st.selectbox(
            "Filter field",
            filter_columns,
            disabled=not apply_filter,
        )

    filter_values = (
        dataframe[filter_column]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    with filter_row[2]:
        filter_value = st.selectbox(
            "Match value",
            filter_values,
            disabled=not apply_filter,
        )

    if apply_filter:
        dataframe = dataframe[
            dataframe[filter_column].astype(str)
            == filter_value
        ].copy()

else:
    st.caption(
        "No additional columns are available for filtering."
    )


# ---------------------------------------------------------------------
# Prepare time series
# ---------------------------------------------------------------------

try:
    series = ts.prepare_series(
        dataframe=dataframe,
        date_column=date_column,
        value_column=value_column,
        replace_zeroes=replace_zeroes,
    )
except Exception as error:
    st.error(str(error))
    st.stop()

if len(series) < 5:
    st.error(
        "At least five usable time-series observations are required."
    )
    st.stop()

if input_method == "Currency rates":
    value_axis_label = (
        f"{currency_to} per {currency_from}"
    )
else:
    value_axis_label = value_column

inferred_frequency = pd.infer_freq(series.index)

st.caption(
    f"Source: `{source_label}` | "
    f"Usable observations: `{len(series)}` | "
    f"Inferred frequency: `{inferred_frequency or 'not detected'}`"
)

st.markdown(
    "<p style='font-size: 20px; font-weight: bold; "
    "margin-bottom: 5px;'>Observed time series</p>",
    unsafe_allow_html=True,
)


if input_method == "Currency rates":

    series_label = (
        f"{currency_from}/{currency_to}: "
        f"{len(series)} observations"
    )

elif apply_filter:

    series_label = (
        f"{filter_column} = {filter_value}: "
        f"{len(series)} observations"
    )

else:

    series_label = (
        f"All data: {len(series)} observations"
    )


initial_figure = ts.make_series_plot(
    series=series,
    value_name=value_axis_label,
    label=series_label,
)

st.pyplot(
    initial_figure,
    use_container_width=False,
)

plt.close(initial_figure)


# ---------------------------------------------------------------------
# Forecast controls
# ---------------------------------------------------------------------

st.divider()

frequency_names = list(
    ts.FREQUENCY_CODES.keys()
)

default_frequency = ts.frequency_name_from_code(
    inferred_frequency
)

if default_frequency not in frequency_names:
    default_frequency = "months (start)"

forecast_row_1 = st.columns(
    [0.9, 1.3, 1.4]
)

with forecast_row_1[0]:

    resample_choice = st.selectbox(
        "Resample dates?",
        ["No", "Yes"],
        index=0,
    )

with forecast_row_1[1]:

    frequency_name = st.selectbox(
        "Required time intervals",
        frequency_names,
        index=frequency_names.index(
            default_frequency
        ),
    )

with forecast_row_1[2]:

    algorithm = st.selectbox(
        "Forecast algorithm",
        [
            "Simple exponential",
            "Double exponential",
            "Triple exponential",
            "ARIMA",
            "Prophet",
        ],
        index=2,
    )

maximum_horizon = ts.maximum_horizon(
    frequency_name
)

forecast_row_2 = st.columns(
    [1.4, 1.0, 1.3]
)

with forecast_row_2[0]:

    forecast_length = st.slider(
        f"Forecast length in {frequency_name}",
        min_value=1,
        max_value=maximum_horizon,
        value=min(12, maximum_horizon),
        step=1,
    )

with forecast_row_2[1]:

    forecast_or_test = st.selectbox(
        "Forecast or back-test?",
        [
            "Forecast",
            "Test",
        ],
        index=0,
    )

holiday_country = "None"

with forecast_row_2[2]:

    if algorithm == "Prophet":

        holiday_country = st.selectbox(
            "Include national holidays",
            ts.PROPHET_COUNTRIES,
            index=0,
        )

    else:

        st.selectbox(
            "Include national holidays",
            ["Only available for Prophet"],
            disabled=True,
        )


# ---------------------------------------------------------------------
# Resample and validate
# ---------------------------------------------------------------------

try:
    prepared_series = ts.resample_series(
        series=series,
        resample=(
            resample_choice == "Yes"
        ),
        frequency_name=frequency_name,
    )
except Exception as error:
    st.error(
        f"Could not prepare the selected frequency: {error}"
    )
    st.stop()

if len(prepared_series) <= forecast_length + 2:
    st.error(
        "The forecast horizon is too long for the available data. "
        "Reduce the forecast length or use a larger dataset."
    )
    st.stop()

available_zoom_dates = list(
    prepared_series.index
)

default_zoom_index = max(
    0,
    len(available_zoom_dates)
    - max(60, forecast_length * 4),
)

# ---------------------------------------------------------------------
# Final plot zoom
# ---------------------------------------------------------------------

st.markdown(
    "<p style='font-size: 20px; font-weight: bold; "
    "margin-bottom: 5px;'>Final plot zoom</p>",
    unsafe_allow_html=True,
)

maximum_history = len(
    prepared_series
)

default_history = min(
    maximum_history,
    max(
        60,
        int(forecast_length) * 4,
    ),
)

zoom_row = st.columns(
    [2.2, 1.0]
)

with zoom_row[0]:

    history_length = st.slider(
        "Historical observations displayed",
        min_value=1,
        max_value=maximum_history,
        value=default_history,
        step=1,
        key="history_length_slider",
    )

    zoom_start = prepared_series.index[
        -history_length
    ]

with zoom_row[1]:

    y_axis_mode = st.selectbox(
        "Y-axis range",
        [
            "Automatic",
            "Manual",
        ],
        index=0,
    )

manual_y_min = None
manual_y_max = None

if y_axis_mode == "Manual":

    visible_values = prepared_series.loc[
        zoom_start:
    ]

    default_y_min = float(
        visible_values.min()
    )

    default_y_max = float(
        visible_values.max()
    )

    if default_y_min == default_y_max:
        default_y_min -= 0.5
        default_y_max += 0.5

    y_range_columns = st.columns(2)

    with y_range_columns[0]:

        manual_y_min = st.number_input(
            "Minimum y-value",
            value=default_y_min,
            format="%.6f",
        )

    with y_range_columns[1]:

        manual_y_max = st.number_input(
            "Maximum y-value",
            value=default_y_max,
            format="%.6f",
        )

    if manual_y_min >= manual_y_max:

        st.error(
            "The minimum y-value must be lower "
            "than the maximum y-value."
        )

        st.stop()

# ---------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------

current_settings = {
    "source": source_label,
    "date_column": date_column,
    "value_column": value_column,
    "replace_zeroes": replace_zeroes,
    "filter_applied": apply_filter,
    "frequency_name": frequency_name,
    "resample": resample_choice,
    "algorithm": algorithm,
    "forecast_length": int(forecast_length),
    "mode": forecast_or_test,
    "holiday_country": holiday_country,
    "series_start": str(prepared_series.index.min()),
    "series_end": str(prepared_series.index.max()),
    "series_length": len(prepared_series),
}

run_forecast = st.button(
    "Run forecast",
    type="primary",
)

if run_forecast:

    with st.spinner(
        f"Running {algorithm} model..."
    ):

        try:
            result = ts.run_forecast(
                series=prepared_series,
                algorithm=algorithm,
                forecast_length=int(
                    forecast_length
                ),
                frequency_name=frequency_name,
                mode=forecast_or_test,
                holiday_country=holiday_country,
            )

            st.session_state[
                "forecast_result"
            ] = result

            st.session_state[
                "forecast_settings"
            ] = current_settings.copy()

        except ImportError as error:
            st.error(str(error))
            st.stop()

        except Exception as error:
            st.error(
                f"Model error: {error}"
            )
            st.stop()

if "forecast_result" not in st.session_state:

    st.info(
        "Choose the model settings above, then to run or re-run select "
        "**Run forecast**."
    )
    st.stop()

saved_settings = st.session_state.get(
    "forecast_settings",
    {},
)

if saved_settings != current_settings:

    st.warning(
        "The data or model settings have changed. "
        "The plot below still shows the last completed model run. "
        "Select **Run forecast** to update the forecast."
    )

result = st.session_state[
    "forecast_result"
]


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

st.markdown(
    "<p style='font-size: 20px; font-weight: bold; "
    "margin-bottom: 5px;'>Forecast results</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"##### Model used: {result['model_label']}"
)

st.markdown(
    f"##### Forecast horizon: "
    f"{forecast_length} {frequency_name}"
)

metric_columns = st.columns(2)


if forecast_or_test == "Test":

    if result["npd"] < 5:

        colour = "#1e7e34"
        icon = "✅"
        accuracy = "Excellent"

    elif result["npd"] < 10:

        colour = "#198754"
        icon = "🟢"
        accuracy = "Good"

    elif result["npd"] < 20:

        colour = "#d39e00"
        icon = "🟡"
        accuracy = "Reasonable"

    else:

        colour = "#c82333"
        icon = "🔴"
        accuracy = "Poor"

    st.markdown(
        f"""
<div style="
padding:18px;
border-radius:10px;
border-left:8px solid {colour};
background:#f8f9fa;
margin-top:10px;
margin-bottom:20px;
">

<div style="
font-size:22px;
font-weight:700;
color:{colour};
margin-bottom:8px;
">
{icon} {accuracy} forecasting accuracy
</div>

<div style="font-size:18px;">
<b>RMSE:</b> {result['rmse']:.4f}<br>
<b>NPD:</b> {result['npd']:.2f}%
</div>

</div>
""",
        unsafe_allow_html=True,
    )

else:

    metric_columns = st.columns(2)

    metric_columns[0].metric(
        "Last observed value",
        f"{prepared_series.iloc[-1]:.4f}",
    )

    metric_columns[1].metric(
        "Final forecast value",
        f"{result['forecast'].iloc[-1]:.4f}",
    )



   

forecast_figure = ts.make_forecast_plot(
    full_series=prepared_series,
    train_series=result["train"],
    fitted_series=result["fitted"],
    forecast_series=result["forecast"],
    actual_test=result["actual_test"],
    model_label=result["model_label"],
    mode=forecast_or_test,
    frequency_name=frequency_name,
    zoom_start=zoom_start,
    #zoom_end=zoom_end,
    value_name=value_axis_label,
    rmse=result["rmse"],
    npd=result["npd"],
    manual_y_min=manual_y_min,
    manual_y_max=manual_y_max,
)

st.pyplot(
    forecast_figure,
    use_container_width=False,
)

plt.close(forecast_figure)

results_frame = pd.DataFrame(
    {
        "Date": result["forecast"].index,
        "Forecast": result["forecast"].values,
    }
)

if (
    forecast_or_test == "Test"
    and result["actual_test"] is not None
):

    results_frame["Actual"] = (
        result["actual_test"]
        .reindex(result["forecast"].index)
        .values
    )

    results_frame["Error"] = (
        results_frame["Forecast"]
        - results_frame["Actual"]
    )

with st.expander(
    "Forecast values",
    expanded=False,
):

    st.dataframe(
        results_frame,
        hide_index=True,
        use_container_width=True,
    )

st.download_button(
    "Download forecast results as CSV",
    data=results_frame.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="time_series_forecast.csv",
    mime="text/csv",
)

with st.expander(
    "Interpretation"
):

    st.markdown(
        """
- **Forecast mode** - the coloured line shows the model's predictions beyond the observed data.
    Compare different forecasting methods to see how sensitive future predictions are to the choice
        of model.
- **Back-test mode** - the model is fitted using only the historical data before the shaded region.
        The hidden observations are then compared with the model's predictions.
        
- **RMSE** measures the average prediction error in the same units as the original data. Lower values indicate more accurate forecasts.
- **NPD** is the  normalised percentage difference divides RMSE by the mean absolute
  observed value over the test period.
"""
    )

with st.expander(
    "Methods"
):

    st.markdown(
        """
- **Simple exponential smoothing** is intended for data without a clear
  trend or seasonal pattern.
- **Double exponential smoothing** adds a linear trend.
- **Triple exponential smoothing** adds both trend and seasonality.
- **ARIMA** uses automated order selection through `pmdarima`.
- **Prophet** is designed for changing trends, missing observations and
  optional national-holiday effects.
"""
    )


    
st.caption(
    f"Data directory: `{DATA_DIR}`"
)
