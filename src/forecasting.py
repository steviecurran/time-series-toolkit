from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import (
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)


FREQUENCY_CODES = {
    "days": "D",
    "business days": "B",
    "weeks": "W",
    "months (start)": "MS",
    "months (end)": "ME",
    "years": "YS",
}

SEASONAL_PERIODS = {
    "days": 7,
    "business days": 5,
    "weeks": 52,
    "months (start)": 12,
    "months (end)": 12,
    "years": 10,
}

PROPHET_COUNTRY_CODES = {
    "None": None,
    "United States": "US",
    "China": "CN",
    "Germany": "DE",
    "Japan": "JP",
    "India": "IN",
    "United Kingdom": "UK",
    "France": "FR",
    "Italy": "IT",
    "Russia": "RU",
    "Canada": "CA",
    "Brazil": "BR",
    "Spain": "ES",
    "Mexico": "MX",
    "South Korea": "KR",
    "Australia": "AU",
}

PROPHET_COUNTRIES = list(
    PROPHET_COUNTRY_CODES.keys()
)


def available_data_files(
    data_directory: Path,
) -> list[Path]:
    """List supported time-series files."""

    if not data_directory.exists():
        return []

    supported_extensions = {
        ".csv",
        ".dat",
        ".txt",
        ".tsv",
    }

    return sorted(
        path
        for path in data_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in supported_extensions
    )


def read_data_file(
    path_or_buffer,
) -> pd.DataFrame:
    """Read CSV or delimited text data."""

    return pd.read_csv(
        path_or_buffer,
        sep=None,
        engine="python",
        comment="#",
    )


def download_currency_data(
    currency_from: str,
    currency_to: str,
    days_back: int,
) -> pd.DataFrame:
    """Download daily exchange-rate data through yfinance."""

    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            "Currency downloads require `yfinance`. "
            "Install it with `pip install yfinance`."
        ) from error

    symbol = (
        f"{currency_from}{currency_to}=X"
    )

    end_date = pd.Timestamp.now(
        tz=None
    ).normalize()

    start_date = (
        end_date
        - pd.Timedelta(days=days_back)
    )

    downloaded = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if downloaded.empty:
        raise ValueError(
            "No currency data were returned."
        )

    close_data = downloaded["Close"]

    if isinstance(
        close_data,
        pd.DataFrame,
    ):
        close_data = close_data.iloc[:, 0]

    return pd.DataFrame(
        {
            "Date": close_data.index,
            "Close": close_data.values,
        }
    )


def prepare_series(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    replace_zeroes: bool = False,
) -> pd.Series:
    """Convert selected columns into a clean chronological series."""

    working = dataframe[
        [
            date_column,
            value_column,
        ]
    ].copy()

    if (
        "year" in date_column.lower()
        and pd.api.types.is_numeric_dtype(
            working[date_column]
        )
    ):
        years = pd.to_numeric(
            working[date_column],
            errors="coerce",
        )

        working[date_column] = pd.to_datetime(
            years.astype("Int64").astype(str),
            format="%Y",
            errors="coerce",
        )

    else:

        working[date_column] = pd.to_datetime(
            working[date_column],
            errors="coerce",
        )

    working[value_column] = pd.to_numeric(
        working[value_column],
        errors="coerce",
    )

    if replace_zeroes:
        working[value_column] = (
            working[value_column]
            .replace(0, np.nan)
        )

    working = working.dropna(
        subset=[
            date_column,
            value_column,
        ]
    )

    working = working.sort_values(
        date_column
    )

    if working[date_column].duplicated().any():

        working = (
            working.groupby(
                date_column,
                as_index=False,
            )[value_column]
            .mean()
        )

    series = working.set_index(
        date_column
    )[value_column].astype(float)

    series.index = pd.DatetimeIndex(
        series.index
    ).tz_localize(None)

    series.name = value_column

    return series


def frequency_name_from_code(
    frequency_code: Optional[str],
) -> Optional[str]:
    """Translate a pandas frequency code into a UI label."""

    if frequency_code is None:
        return None

    normalised = frequency_code.upper()

    if normalised.startswith("B"):
        return "business days"

    if normalised.startswith("D"):
        return "days"

    if normalised.startswith("W"):
        return "weeks"

    if normalised.startswith("MS"):
        return "months (start)"

    if (
        normalised.startswith("M")
        or normalised.startswith("ME")
    ):
        return "months (end)"

    if (
        normalised.startswith("Y")
        or normalised.startswith("A")
    ):
        return "years"

    return None


def maximum_horizon(
    frequency_name: str,
) -> int:
    """Return a practical UI maximum forecast horizon."""

    if frequency_name in [
        "days",
        "business days",
    ]:
        return 31

    if frequency_name == "weeks":
        return 52

    if frequency_name in [
        "months (start)",
        "months (end)",
    ]:
        return 24

    return 100


def resample_series(
    series: pd.Series,
    resample: bool,
    frequency_name: str,
) -> pd.Series:
    """Optionally resample and interpolate to a regular interval."""

    if not resample:
        return series.copy()

    frequency_code = FREQUENCY_CODES[
        frequency_name
    ]

    result = series.resample(
        frequency_code
    ).mean()

    result = result.interpolate(
        method="time"
    )

    result = result.ffill().bfill()

    return result


def _future_index(
    last_date: pd.Timestamp,
    length: int,
    frequency_name: str,
) -> pd.DatetimeIndex:
    """Create a forecast date index."""

    frequency_code = FREQUENCY_CODES[
        frequency_name
    ]

    return pd.date_range(
        start=last_date,
        periods=length + 1,
        freq=frequency_code,
    )[1:]


def _fit_exponential(
    train: pd.Series,
    algorithm: str,
    forecast_length: int,
    frequency_name: str,
) -> tuple[pd.Series, pd.Series, str]:
    """Fit one of the exponential-smoothing methods."""

    if algorithm == "Simple exponential":

        model = SimpleExpSmoothing(
            train,
            initialization_method="estimated",
        )

        fitted_model = model.fit(
            optimized=True
        )

        model_name = "Simple exponential"

    elif algorithm == "Double exponential":

        model = Holt(
            train,
            initialization_method="estimated",
        )

        fitted_model = model.fit(
            optimized=True
        )

        model_name = "Double exponential"

    else:

        seasonal_periods = SEASONAL_PERIODS[
            frequency_name
        ]

        if len(train) < seasonal_periods * 2:
            raise ValueError(
                "Triple exponential smoothing requires "
                f"at least {seasonal_periods * 2} "
                "training observations for this frequency."
            )

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )

        fitted_model = model.fit(
            optimized=True
        )

        model_name = "Triple exponential"

    fitted = pd.Series(
        fitted_model.fittedvalues,
        index=train.index,
        name="Fitted",
    )

    forecast_values = (
        fitted_model.forecast(
            forecast_length
        )
    )

    if not isinstance(
        forecast_values.index,
        pd.DatetimeIndex,
    ):

        forecast_values.index = _future_index(
            train.index[-1],
            forecast_length,
            frequency_name,
        )

    forecast = pd.Series(
        np.asarray(forecast_values),
        index=forecast_values.index,
        name="Forecast",
    )

    smoothing_level = (
        fitted_model.params.get(
            "smoothing_level",
            np.nan,
        )
    )

    label = (
        f"{model_name}: "
        f"α = {smoothing_level:.4f}"
    )

    return fitted, forecast, label


def _fit_arima(
    train: pd.Series,
    forecast_length: int,
    frequency_name: str,
) -> tuple[pd.Series, pd.Series, str]:
    """Fit seasonal auto-ARIMA."""

    try:
        from pmdarima import auto_arima
    except ImportError as error:
        raise ImportError(
            "ARIMA requires `pmdarima`. "
            "Install it with `pip install pmdarima`."
        ) from error

    seasonal_period = (
        SEASONAL_PERIODS[
            frequency_name
        ]
    )

    use_seasonality = (
        seasonal_period > 1
        and len(train)
        >= seasonal_period * 2
    )

    model = auto_arima(
        train,
        seasonal=use_seasonality,
        m=(
            seasonal_period
            if use_seasonality
            else 1
        ),
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    fitted = pd.Series(
        model.predict_in_sample(),
        index=train.index,
        name="Fitted",
    )

    forecast_index = _future_index(
        train.index[-1],
        forecast_length,
        frequency_name,
    )

    forecast = pd.Series(
        model.predict(
            n_periods=forecast_length
        ),
        index=forecast_index,
        name="Forecast",
    )

    order = model.order
    seasonal_order = model.seasonal_order

    if seasonal_order[-1] > 0:

        label = (
            f"ARIMA {order}"
            f"{seasonal_order}"
        )

    else:

        label = f"ARIMA {order}"

    return fitted, forecast, label


def _fit_prophet(
    train: pd.Series,
    forecast_length: int,
    frequency_name: str,
    holiday_country: str,
) -> tuple[pd.Series, pd.Series, str]:
    """Fit Prophet with optional national holidays."""

    try:
        from prophet import Prophet
    except ImportError as error:
        raise ImportError(
            "Prophet requires the optional `prophet` package. "
            "Install it with `pip install prophet`."
        ) from error

    prophet_data = pd.DataFrame(
        {
            "ds": train.index,
            "y": train.values,
        }
    )

    model = Prophet()

    country_code = (
        PROPHET_COUNTRY_CODES[
            holiday_country
        ]
    )

    if country_code is not None:

        model.add_country_holidays(
            country_name=country_code
        )

    model.fit(prophet_data)

    frequency_code = FREQUENCY_CODES[
        frequency_name
    ]

    future = model.make_future_dataframe(
        periods=forecast_length,
        freq=frequency_code,
    )

    predicted = model.predict(future)

    fitted = pd.Series(
        predicted["yhat"]
        .iloc[:len(train)]
        .values,
        index=train.index,
        name="Fitted",
    )

    forecast_dates = pd.to_datetime(
        predicted["ds"]
        .iloc[-forecast_length:]
        .values
    )

    forecast = pd.Series(
        predicted["yhat"]
        .iloc[-forecast_length:]
        .values,
        index=forecast_dates,
        name="Forecast",
    )

    return fitted, forecast, "Prophet"


def run_forecast(
    series: pd.Series,
    algorithm: str,
    forecast_length: int,
    frequency_name: str,
    mode: str,
    holiday_country: str = "None",
) -> dict:
    """Fit a model and optionally backtest it."""

    if mode == "Test":

        train = series.iloc[
            :-forecast_length
        ].copy()

        actual_test = series.iloc[
            -forecast_length:
        ].copy()

    else:

        train = series.copy()
        actual_test = None

    if len(train) < 3:
        raise ValueError(
            "Too few training observations remain."
        )

    if algorithm in [
        "Simple exponential",
        "Double exponential",
        "Triple exponential",
    ]:

        fitted, forecast, model_label = (
            _fit_exponential(
                train=train,
                algorithm=algorithm,
                forecast_length=forecast_length,
                frequency_name=frequency_name,
            )
        )

    elif algorithm == "ARIMA":

        fitted, forecast, model_label = (
            _fit_arima(
                train=train,
                forecast_length=forecast_length,
                frequency_name=frequency_name,
            )
        )

    else:

        fitted, forecast, model_label = (
            _fit_prophet(
                train=train,
                forecast_length=forecast_length,
                frequency_name=frequency_name,
                holiday_country=holiday_country,
            )
        )

    rmse = np.nan
    npd = np.nan

    if actual_test is not None:

        forecast = forecast.copy()
        forecast.index = actual_test.index

        actual_values = actual_test.values
        forecast_values = forecast.values

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        actual_values
                        - forecast_values
                    ) ** 2
                )
            )
        )

        mean_actual = float(
            np.mean(
                np.abs(actual_values)
            )
        )

        if mean_actual != 0:

            npd = (
                100.0
                * rmse
                / mean_actual
            )

    return {
        "train": train,
        "actual_test": actual_test,
        "fitted": fitted,
        "forecast": forecast,
        "model_label": model_label,
        "rmse": rmse,
        "npd": npd,
    }


def make_series_plot(
    series: pd.Series,
    value_name: str,
    label: str,
):
    """Create the initial notebook-style time-series plot."""

    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    plt.setp(
        axis.spines.values(),
        linewidth=2,
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=6,
        width=1.5,
        which="major",
        right=True,
        top=True,
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=3,
        width=1.5,
        which="minor",
        right=True,
        top=True,
    )

    axis.plot(
        series.index,
        series.values,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=label,
    )

    axis.set_xlabel(
        "Date",
        fontsize=12,
    )

    axis.set_ylabel(
        value_name,
        fontsize=12,
    )

    axis.tick_params(
        axis="both",
        labelsize=12,
    )

    axis.legend(
        fontsize=10,
        loc="best",
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    return figure


def make_forecast_plot(
    full_series: pd.Series,
    train_series: pd.Series,
    fitted_series: pd.Series,
    forecast_series: pd.Series,
    actual_test: Optional[pd.Series],
    model_label: str,
    mode: str,
    frequency_name: str,
    zoom_start: pd.Timestamp,
    value_name: str,
    rmse: float,
    npd: float,
    manual_y_min: Optional[float] = None,
    manual_y_max: Optional[float] = None,
):
    """Create the notebook-style fitted and forecast plot."""

    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    plt.setp(
        axis.spines.values(),
        linewidth=2,
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=6,
        width=1.5,
        which="major",
        right=True,
        top=True,
    )

    axis.tick_params(
        direction="in",
        pad=7,
        length=3,
        width=1.5,
        which="minor",
        right=True,
        top=True,
    )

    axis.plot(
        full_series.index,
        full_series.values,
        label="Actual data",
        color="black",
        linewidth=1.5,
        zorder=1,
    )

    axis.plot(
        fitted_series.index,
        fitted_series.values,
        linestyle="--",
        color="blue",
        linewidth=1.5,
        label=model_label,
        zorder=2,
    )

    forecast_label = (
        f"{len(forecast_series)} "
        f"{frequency_name} forecast period"
    )

    if mode == "Test":

        forecast_label += (
            f" ⇒ RMSE = {rmse:.2f}, "
            f"NPD = {npd:.2f}%"
        )

    axis.plot(
        forecast_series.index,
        forecast_series.values,
        color="red",
        linestyle="dotted",
        linewidth=2,
        label=forecast_label,
        zorder=3,
    )

    if actual_test is not None:

        axis.scatter(
            actual_test.index,
            actual_test.values,
            color="black",
            s=22,
            label="Held-out actual values",
            zorder=4,
        )

    axis.axvspan(
        forecast_series.index[0],
        forecast_series.index[-1],
        color="silver",
        alpha=0.3,
        zorder=0,
    )

    if actual_test is not None:
        display_end = pd.Timestamp(actual_test.index[-1])

    else:
        display_end = pd.Timestamp(forecast_series.index[-1])
    
    axis.set_xlim(
        left=pd.Timestamp(
            zoom_start
        ),
        right=display_end,
    )

    visible_actual = full_series[
        (
            full_series.index
            >= pd.Timestamp(zoom_start)
        )
        & (
            full_series.index
            <= display_end
        )
    ]

    values_for_range = [
        visible_actual.values,
        forecast_series.values,
    ]

    if actual_test is not None:
        values_for_range.append(
            actual_test.values
        )

    combined_values = np.concatenate(
        values_for_range
    )

    if (
        manual_y_min is not None
        and manual_y_max is not None
    ):

        axis.set_ylim(
            manual_y_min,
            manual_y_max,
        )

    elif len(combined_values) > 0:

        minimum = np.nanmin(
            combined_values
        )

        maximum = np.nanmax(
            combined_values
        )

        padding = (
            (maximum - minimum)
            * 0.1
        )

        if padding == 0:
            padding = max(
                abs(maximum) * 0.05,
                0.5,
            )

        axis.set_ylim(
            minimum - padding,
            maximum + padding,
        )

    axis.set_xlabel(
        "Date",
        fontsize=12,
    )

    axis.set_ylabel(
        value_name,
        fontsize=12,
    )

    axis.tick_params(
        axis="both",
        labelsize=12,
    )

    axis.legend(
        fontsize=9.5,
        loc="upper left",
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    return figure
