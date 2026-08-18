"""Time-series forecasting over a dataset artifact (statsforecast AutoETS —
no torch, no CmdStan)."""

import anyio


def _forecast_sync(
    columns: list[dict], rows: list[list], date_column: str, value_column: str,
    horizon: int, freq: str,
):
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    names = [c["name"] for c in columns]
    frame = pd.DataFrame(rows, columns=names)
    if date_column not in names or value_column not in names:
        raise ValueError(
            f"colunas '{date_column}'/'{value_column}' não existem no dataset ({names})"
        )
    frame = frame[[date_column, value_column]].dropna()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame[value_column] = pd.to_numeric(frame[value_column])
    frame = frame.sort_values(date_column)
    if len(frame) < 6:
        raise ValueError("mínimo de 6 pontos históricos para prever")

    data = frame.rename(columns={date_column: "ds", value_column: "y"})
    data["unique_id"] = "serie"
    model = StatsForecast(models=[AutoETS(season_length=_season(freq))], freq=freq)
    prediction = model.forecast(df=data, h=horizon)
    out = prediction.rename(columns={"AutoETS": "previsao"})
    history = [
        [d.strftime("%Y-%m-%d"), float(v)]
        for d, v in zip(data["ds"], data["y"], strict=True)
    ]
    future = [
        [d.strftime("%Y-%m-%d"), round(float(v), 4)]
        for d, v in zip(out["ds"], out["previsao"], strict=True)
    ]
    return history, future


def _season(freq: str) -> int:
    return {"M": 12, "MS": 12, "W": 52, "D": 7, "Q": 4, "QS": 4, "Y": 1}.get(freq, 1)


async def forecast_series(
    columns: list[dict], rows: list[list], date_column: str, value_column: str,
    horizon: int = 6, freq: str = "MS",
):
    """(history, future) — each a list of [iso_date, value]."""
    return await anyio.to_thread.run_sync(
        lambda: _forecast_sync(columns, rows, date_column, value_column, horizon, freq)
    )
