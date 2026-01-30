from dagster import Definitions, load_assets_from_modules

from src.stochastic_weather.assets import ingest, validate, analysis, uncertainty

all_assets = load_assets_from_modules([ingest, validate, analysis, uncertainty])

defs = Definitions(
    assets=all_assets,
)
