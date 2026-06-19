#!/usr/bin/env python3
"""
Open Resilience Uganda: real-data macro--micro climate resilience pipeline.

This script is designed as a replication-grade evidence generator, not as a
loose plotting script. It builds district-year climate resilience evidence from
public data, produces manuscript-ready tables/figures, and separates main-text
outputs from appendix/supplementary audit files.

IMPORTANT EVIDENCE RULES
------------------------
1. Empirical outputs are produced from real public data or user-supplied
   controlled-access validation files. Synthetic data are not used for evidence.
2. Google Earth Engine is used for gridded public products: CHIRPS rainfall,
   MODIS NDVI/EVI, VIIRS Black Marble, and WorldPop population.
3. World Bank WDI is downloaded through the public World Bank API.
4. FAOSTAT/LSMS/DHS validation files are optional and must be supplied by the
   user when license or access rules require controlled access.
5. All outputs are written as CSV, LaTeX, PNG, GeoJSON, and audit logs so that a
   social scientist or industry analyst can inspect each step.

Typical use
-----------
    earthengine authenticate
    python scripts/09_reproduce_all.py --config configs/uganda.yaml

For machines without Earth Engine access, first prepare a real extracted CSV at:
    data/interim/district_year_remote_sensing.csv
with columns:
    district_id, district_name, year, rain_season, rain_annual, ndvi, evi,
    nightlights, population
then run:
    python scripts/09_reproduce_all.py --config configs/uganda.yaml --skip-gee

Author: Ronald Katende workflow support
License: MIT-compatible, adapt to repository license.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

# Heavy geospatial/statistical imports. The script checks these early and gives
# clear installation hints rather than failing deep inside the pipeline.
try:
    import geopandas as gpd
    from shapely.geometry import mapping
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "This pipeline requires geopandas and shapely. Install with: "
        "conda install -c conda-forge geopandas shapely"
    ) from exc

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    outputs: Path
    main_tables: Path
    appendix_tables: Path
    figures: Path
    maps: Path
    logs: Path


def setup_paths(config: Dict[str, Any]) -> Paths:
    root = Path.cwd()
    data = root / str(config.get("data_dir", "data"))
    outputs = root / str(config.get("outputs_dir", "outputs"))
    paths = Paths(
        root=root,
        data=data,
        raw=data / "raw",
        interim=data / "interim",
        processed=data / "processed",
        outputs=outputs,
        main_tables=outputs / "main_text" / "tables",
        appendix_tables=outputs / "appendix" / "tables",
        figures=outputs / "main_text" / "figures",
        maps=outputs / "main_text" / "maps",
        logs=outputs / "logs",
    )
    for p in paths.__dict__.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return paths


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(paths: Paths) -> None:
    log_path = paths.logs / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )
    logging.info("Writing log to %s", log_path)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_table(df: pd.DataFrame, csv_path: Path, latex_path: Optional[Path] = None, index: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    if latex_path is not None:
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            latex_path.write_text(df.to_latex(index=index, escape=True), encoding="utf-8")
        except Exception as exc:
            logging.warning("Could not write LaTeX table %s: %s", latex_path, exc)


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s).strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Data source acquisition
# ---------------------------------------------------------------------------

def download_geoboundaries(country_iso3: str, admin_level: str, out_dir: Path, user_boundary_path: Optional[str]) -> Path:
    """Return a local boundary file path.

    Priority:
    1. Use user-supplied boundary file if provided.
    2. Use cached geoBoundaries GeoJSON.
    3. Download geoBoundaries Open boundary from public API.

    The boundary file should be fixed for the analysis to avoid administrative
    splits being misread as climate or capacity changes.
    """
    if user_boundary_path:
        p = Path(user_boundary_path)
        if p.exists():
            logging.info("Using user boundary file: %s", p)
            return p
        raise FileNotFoundError(f"Configured boundary_path does not exist: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / f"geoboundaries_{country_iso3}_{admin_level}.geojson"
    if cached.exists():
        logging.info("Using cached boundary file: %s", cached)
        return cached

    api_url = f"https://www.geoboundaries.org/api/current/gbOpen/{country_iso3}/{admin_level}/"
    logging.info("Downloading boundary metadata from %s", api_url)
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    meta = r.json()
    geojson_url = meta.get("gjDownloadURL") or meta.get("gjDownloadURLSimplified")
    if not geojson_url:
        raise RuntimeError(f"Could not find GeoJSON URL in geoBoundaries API response for {country_iso3}-{admin_level}.")
    logging.info("Downloading boundary GeoJSON from %s", geojson_url)
    gj = requests.get(geojson_url, timeout=120)
    gj.raise_for_status()
    cached.write_bytes(gj.content)
    return cached


def load_boundaries(path: Path, config: Dict[str, Any]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Boundary file loaded but contains no features.")
    if gdf.crs is None:
        logging.warning("Boundary CRS is missing; assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    id_col = pick_first_existing(gdf.columns, config.get("boundary_id_candidates", []))
    name_col = pick_first_existing(gdf.columns, config.get("boundary_name_candidates", []))
    if id_col is None:
        id_col = "district_id"
        gdf[id_col] = [f"D{i:03d}" for i in range(len(gdf))]
    if name_col is None:
        name_col = id_col

    gdf = gdf[[id_col, name_col, "geometry"]].copy()
    gdf = gdf.rename(columns={id_col: "district_id", name_col: "district_name"})
    gdf["district_id"] = gdf["district_id"].astype(str)
    gdf["district_name"] = gdf["district_name"].astype(str)
    gdf["area_km2"] = gdf.to_crs(3857).geometry.area / 1e6
    logging.info("Loaded %d district/administrative units.", len(gdf))
    return gdf


def pick_first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def download_wdi(iso3_list: Sequence[str], indicators: Dict[str, str], start_year: int, end_year: int, out_dir: Path) -> pd.DataFrame:
    """Download World Bank WDI indicators via public API."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"wdi_{start_year}_{end_year}_{'_'.join(iso3_list)}.csv"
    if cache_path.exists():
        logging.info("Using cached WDI file: %s", cache_path)
        return pd.read_csv(cache_path)

    rows: List[Dict[str, Any]] = []
    countries = ";".join(iso3_list)
    for code, short_name in indicators.items():
        url = (
            f"https://api.worldbank.org/v2/country/{countries}/indicator/{code}"
            f"?format=json&per_page=20000&date={start_year}:{end_year}"
        )
        logging.info("Downloading WDI indicator %s (%s)", code, short_name)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            logging.warning("No WDI data returned for %s", code)
            continue
        for rec in payload[1]:
            rows.append(
                {
                    "iso3": rec.get("countryiso3code"),
                    "country": rec.get("country", {}).get("value"),
                    "year": int(rec.get("date")),
                    "indicator_code": code,
                    "indicator": short_name,
                    "value": rec.get("value"),
                }
            )
        time.sleep(0.2)

    long = pd.DataFrame(rows)
    if long.empty:
        raise RuntimeError("World Bank API returned no data for configured indicators.")
    wide = long.pivot_table(index=["iso3", "country", "year"], columns="indicator", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide.to_csv(cache_path, index=False)
    logging.info("Saved WDI data to %s", cache_path)
    return wide


def load_optional_faostat(config: Dict[str, Any], paths: Paths) -> Optional[pd.DataFrame]:
    """Load optional FAOSTAT validation CSV.

    We do not bundle FAOSTAT-derived files automatically because users may want
    different crops/items/elements. This function accepts a user-provided CSV
    and standardizes common column names. The README tells users how to provide
    the file. The rest of the pipeline runs without it, but the validation audit
    will mark agriculture validation as partial.
    """
    p = config.get("faostat_csv")
    if not p:
        logging.info("No FAOSTAT CSV configured; national crop validation will be skipped.")
        return None
    path = Path(p)
    if not path.exists():
        logging.warning("Configured FAOSTAT CSV does not exist: %s; skipping.", path)
        return None
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in {"area", "country", "area item"}:
            rename[col] = "country"
        elif c in {"year", "year code"}:
            rename[col] = "year"
        elif c in {"item", "item name"}:
            rename[col] = "item"
        elif c in {"element", "element name"}:
            rename[col] = "element"
        elif c in {"value", "obs_value"}:
            rename[col] = "value"
    df = df.rename(columns=rename)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.to_csv(paths.raw / "faostat_user_supplied_standardized.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Google Earth Engine zonal statistics
# ---------------------------------------------------------------------------

def init_earth_engine(config: Dict[str, Any]) -> Any:
    """Initialize Earth Engine and return the ee module."""
    try:
        import ee
    except Exception as exc:
        raise ImportError(
            "earthengine-api is required for direct real-data extraction. "
            "Install with conda install -c conda-forge earthengine-api geemap."
        ) from exc

    gee = config.get("gee", {})
    service_account = gee.get("service_account")
    key_file = gee.get("key_file")
    try:
        if service_account and key_file:
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        logging.info("Initialized Google Earth Engine.")
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Earth Engine. Run `earthengine authenticate` "
            "or configure a service account in configs/uganda.yaml."
        ) from exc
    return ee


def gdf_to_ee_feature_collection(gdf: gpd.GeoDataFrame, ee: Any) -> Any:
    """Convert a GeoDataFrame to an Earth Engine FeatureCollection.

    Uses plain GeoJSON conversion to avoid requiring geemap at runtime. For very
    complex boundaries, consider simplifying the boundary file before running.
    """
    features = []
    for _, row in gdf.iterrows():
        geom = mapping(row.geometry)
        props = {"district_id": str(row["district_id"]), "district_name": str(row["district_name"])}
        features.append(ee.Feature(ee.Geometry(geom), props))
    return ee.FeatureCollection(features)


def ee_reduce_regions_to_df(image: Any, fc: Any, ee: Any, scale: int, reducer: str = "mean") -> pd.DataFrame:
    if reducer == "mean":
        red = ee.Reducer.mean()
    elif reducer == "sum":
        red = ee.Reducer.sum()
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")
    stats = image.reduceRegions(collection=fc, reducer=red, scale=scale, crs="EPSG:4326", tileScale=4)
    features = stats.getInfo().get("features", [])
    records = [f.get("properties", {}) for f in features]
    return pd.DataFrame(records)


def build_gee_district_year_stats(
    gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    paths: Paths,
    force: bool = False,
) -> pd.DataFrame:
    """Extract district-year CHIRPS/MODIS/VIIRS/WorldPop statistics from GEE."""
    out = paths.interim / "district_year_remote_sensing.csv"
    if out.exists() and not force:
        logging.info("Using cached district-year remote-sensing table: %s", out)
        return pd.read_csv(out)

    ee = init_earth_engine(config)
    fc = gdf_to_ee_feature_collection(gdf, ee)
    gee = config["gee"]["collections"]
    season_months = list(config.get("season_months", [3, 4, 5, 6]))
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    country_iso3 = config.get("country_iso3", "UGA")

    all_rows: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        logging.info("Extracting GEE district stats for %d", year)
        season_start = f"{year}-{min(season_months):02d}-01"
        # End date is first day of month after the last season month.
        last_month = max(season_months)
        if last_month == 12:
            season_end = f"{year+1}-01-01"
        else:
            season_end = f"{year}-{last_month+1:02d}-01"
        year_start, year_end = f"{year}-01-01", f"{year+1}-01-01"

        # CHIRPS rainfall: seasonal and annual total precipitation.
        chirps = ee.ImageCollection(gee["chirps"]["id"]).select(gee["chirps"]["band"])
        rain_season_img = chirps.filterDate(season_start, season_end).sum().rename("rain_season")
        rain_annual_img = chirps.filterDate(year_start, year_end).sum().rename("rain_annual")
        rain_img = rain_season_img.addBands(rain_annual_img)
        rain_df = ee_reduce_regions_to_df(rain_img, fc, ee, scale=int(gee["chirps"].get("scale", 5500)), reducer="mean")

        # MODIS vegetation: mean seasonal NDVI/EVI after scale factor.
        modis_cfg = gee["modis"]
        bands = modis_cfg.get("bands", ["NDVI", "EVI"])
        scale_factor = float(modis_cfg.get("scale_factor", 0.0001))

        def scale_modis(img):
            return img.select(bands).multiply(scale_factor).copyProperties(img, ["system:time_start"])

        modis_img = (
            ee.ImageCollection(modis_cfg["id"])
            .filterDate(season_start, season_end)
            .map(scale_modis)
            .mean()
            .rename([b.lower() for b in bands])
        )
        veg_df = ee_reduce_regions_to_df(modis_img, fc, ee, scale=int(modis_cfg.get("scale", 250)), reducer="mean")

        # VIIRS Black Marble / night lights: unavailable before start_year.
        viirs_cfg = gee.get("viirs", {})
        if year >= int(viirs_cfg.get("start_year", 2012)):
            try:
                viirs_img = (
                    ee.ImageCollection(viirs_cfg["id"])
                    .filterDate(year_start, year_end)
                    .select(viirs_cfg["band"])
                    .mean()
                    .rename("nightlights")
                )
                viirs_df = ee_reduce_regions_to_df(viirs_img, fc, ee, scale=int(viirs_cfg.get("scale", 500)), reducer="mean")
            except Exception as exc:
                logging.warning("VIIRS extraction failed for %d: %s", year, exc)
                viirs_df = rain_df[["district_id", "district_name"]].copy()
                viirs_df["nightlights"] = np.nan
        else:
            viirs_df = rain_df[["district_id", "district_name"]].copy()
            viirs_df["nightlights"] = np.nan

        # WorldPop gridded population: sum over districts. GEE WorldPop images
        # usually carry country/year metadata; if filtering fails, the code
        # records missing values and the audit makes this visible.
        wp_cfg = gee.get("worldpop", {})
        try:
            wp_collection = ee.ImageCollection(wp_cfg["id"]).filterDate(year_start, year_end)
            try:
                wp_collection = wp_collection.filter(ee.Filter.eq("country", country_iso3))
            except Exception:
                pass
            wp_img = wp_collection.select(wp_cfg.get("band", "population")).mosaic().rename("population")
            pop_df = ee_reduce_regions_to_df(wp_img, fc, ee, scale=int(wp_cfg.get("scale", 100)), reducer="sum")
        except Exception as exc:
            logging.warning("WorldPop extraction failed for %d: %s", year, exc)
            pop_df = rain_df[["district_id", "district_name"]].copy()
            pop_df["population"] = np.nan

        # Merge the yearly tables on district identifiers.
        dfs = [rain_df, veg_df, viirs_df, pop_df]
        year_df = dfs[0]
        for other in dfs[1:]:
            keep = [c for c in other.columns if c not in year_df.columns or c in {"district_id", "district_name"}]
            year_df = year_df.merge(other[keep], on=["district_id", "district_name"], how="left")
        year_df["year"] = year
        all_rows.append(year_df)

    remote = pd.concat(all_rows, ignore_index=True)
    # EE reducers may produce columns like NDVI_mean depending on API version.
    remote = normalize_remote_columns(remote)
    remote.to_csv(out, index=False)
    logging.info("Saved GEE district-year remote sensing table: %s", out)
    return remote


def normalize_remote_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        c = col.lower()
        if c in {"rain_season_mean", "mean"} and "rain_season" not in df.columns:
            rename[col] = "rain_season"
        elif c == "rain_annual_mean":
            rename[col] = "rain_annual"
        elif c == "ndvi_mean":
            rename[col] = "ndvi"
        elif c == "evi_mean":
            rename[col] = "evi"
        elif c == "nightlights_mean":
            rename[col] = "nightlights"
        elif c == "population_sum":
            rename[col] = "population"
    df = df.rename(columns=rename)
    # Sometimes reducer properties are named after image bands directly.
    for base in ["rain_season", "rain_annual", "ndvi", "evi", "nightlights", "population"]:
        candidates = [c for c in df.columns if c.lower() == base.lower()]
        if candidates and candidates[0] != base:
            df = df.rename(columns={candidates[0]: base})
    return df


# ---------------------------------------------------------------------------
# Evidence panel construction and indices
# ---------------------------------------------------------------------------

def build_district_year_panel(remote: pd.DataFrame, gdf: gpd.GeoDataFrame, wdi_uganda: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    """Merge remote sensing, boundaries, and WDI context into one panel."""
    panel = remote.copy()
    panel["district_id"] = panel["district_id"].astype(str)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype(int)
    # Add area for density and audit.
    area = gdf[["district_id", "district_name", "area_km2"]].drop_duplicates("district_id")
    panel = panel.merge(area, on=["district_id", "district_name"], how="left")

    # Merge country-level WDI as context. This is clearly marked as national
    # context, not district-level infrastructure.
    wdi = wdi_uganda.copy()
    wdi["year"] = pd.to_numeric(wdi["year"], errors="coerce")
    panel = panel.merge(wdi.drop(columns=["country"], errors="ignore"), on="year", how="left")
    panel["country_iso3"] = config.get("country_iso3", "UGA")
    panel["country_name"] = config.get("country_name", "Uganda")

    # Coerce numeric columns.
    for col in panel.columns:
        if col not in {"district_id", "district_name", "country_iso3", "country_name", "iso3"}:
            panel[col] = pd.to_numeric(panel[col], errors="ignore")

    panel.to_csv(paths.processed / "district_year_raw_panel.csv", index=False)
    return panel


def add_indices(panel: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    """Construct stress, vegetation, capacity, exposure, uncertainty, and priority."""
    df = panel.copy()
    baseline_start = int(config.get("baseline_start", config["start_year"]))
    baseline_end = int(config.get("baseline_end", min(config["end_year"], baseline_start + 14)))
    eps = 1e-8

    # Baselines by district.
    base = df[(df["year"] >= baseline_start) & (df["year"] <= baseline_end)].copy()
    for col in ["rain_season", "rain_annual", "ndvi", "evi"]:
        if col not in df.columns:
            df[col] = np.nan
    base_stats = base.groupby("district_id").agg(
        rain_mu=("rain_season", "mean"),
        rain_sd=("rain_season", "std"),
        ndvi_mu=("ndvi", "mean"),
        ndvi_sd=("ndvi", "std"),
        evi_mu=("evi", "mean"),
        evi_sd=("evi", "std"),
    ).reset_index()
    df = df.merge(base_stats, on="district_id", how="left")

    # Higher = worse rainfall stress. This assumes low rainfall is the main
    # stress. Appendix robustness should test wet-stress variants where relevant.
    df["rain_stress"] = -((df["rain_season"] - df["rain_mu"]) / (df["rain_sd"].replace(0, np.nan) + eps))
    df["ndvi_anomaly"] = (df["ndvi"] - df["ndvi_mu"]) / (df["ndvi_sd"].replace(0, np.nan) + eps)
    df["evi_anomaly"] = (df["evi"] - df["evi_mu"]) / (df["evi_sd"].replace(0, np.nan) + eps)
    df["adverse_vegetation"] = -df["ndvi_anomaly"]

    # Capacity: use district nightlights if available, plus national WDI context.
    capacity_components = []
    if "nightlights" in df.columns:
        df["nightlights_norm"] = minmax_by_year_or_global(df["nightlights"])
        capacity_components.append("nightlights_norm")
    if "electricity_access" in df.columns:
        df["electricity_access_norm"] = minmax_by_year_or_global(df["electricity_access"])
        capacity_components.append("electricity_access_norm")
    if "gdp_pc_constant" in df.columns:
        df["gdp_pc_norm"] = minmax_by_year_or_global(np.log1p(df["gdp_pc_constant"]))
        capacity_components.append("gdp_pc_norm")

    if capacity_components:
        df["capacity_index"] = df[capacity_components].mean(axis=1, skipna=True)
    else:
        df["capacity_index"] = np.nan

    # Population exposure: report both counts and shares. The threshold is set
    # from observed stress distribution and saved in the audit.
    stress_threshold = df["rain_stress"].quantile(0.80)
    df["stress_threshold"] = stress_threshold
    df["exposed_population"] = np.where(df["rain_stress"] > stress_threshold, df.get("population", np.nan), 0)
    df["exposed_population_share"] = np.where(
        pd.to_numeric(df.get("population", np.nan), errors="coerce") > 0,
        df["exposed_population"] / df["population"],
        np.nan,
    )

    # Normalized components for the priority score.
    df["S_norm"] = robust_minmax(df["rain_stress"])
    df["C_norm"] = robust_minmax(df["capacity_index"])
    df["E_norm"] = robust_minmax(df["exposed_population_share"].fillna(0))
    df["A_bad_norm"] = robust_minmax(df["adverse_vegetation"])

    core_cols = ["rain_stress", "ndvi_anomaly", "capacity_index", "exposed_population_share"]
    missing_rate = df[core_cols].isna().mean(axis=1)
    df["missingness_uncertainty"] = missing_rate
    # Ranking instability proxy: districts with fewer valid years are more uncertain.
    valid_years = df.groupby("district_id")["rain_stress"].transform(lambda x: x.notna().sum())
    max_valid = max(1, valid_years.max())
    df["coverage_uncertainty"] = 1 - valid_years / max_valid
    df["U_norm"] = robust_minmax(0.65 * df["missingness_uncertainty"] + 0.35 * df["coverage_uncertainty"])

    w = config.get("priority_weights", {})
    df["priority_score"] = (
        float(w.get("stress", 0.30)) * df["S_norm"].fillna(0)
        + float(w.get("weak_capacity", 0.22)) * (1 - df["C_norm"].fillna(df["C_norm"].median()))
        + float(w.get("exposure", 0.20)) * df["E_norm"].fillna(0)
        + float(w.get("adverse_vegetation", 0.20)) * df["A_bad_norm"].fillna(0)
        + float(w.get("uncertainty", 0.08)) * df["U_norm"].fillna(0)
    )

    thr = config.get("action_thresholds", {})
    high_p = df["priority_score"].quantile(float(thr.get("high_priority_quantile", 0.80)))
    med_p = df["priority_score"].quantile(float(thr.get("medium_priority_quantile", 0.50)))
    high_u = df["U_norm"].quantile(float(thr.get("high_uncertainty_quantile", 0.60)))
    df["action_category"] = np.select(
        [
            (df["priority_score"] >= high_p) & (df["U_norm"] <= high_u),
            (df["priority_score"] >= high_p) & (df["U_norm"] > high_u),
            df["priority_score"] >= med_p,
        ],
        ["Act now", "Measure first", "Monitor"],
        default="Lower priority",
    )

    save_json(
        {
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "rain_stress_threshold_80pct": float(stress_threshold),
            "priority_high_threshold": float(high_p),
            "priority_medium_threshold": float(med_p),
            "uncertainty_high_threshold": float(high_u),
            "capacity_components": capacity_components,
            "priority_weights": w,
        },
        paths.appendix_tables / "index_construction_audit.json",
    )
    df.to_csv(paths.processed / "district_year_indices.csv", index=False)
    return df


def minmax_by_year_or_global(x: pd.Series) -> pd.Series:
    return robust_minmax(pd.to_numeric(x, errors="coerce"))


def robust_minmax(x: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    lo, hi = x.quantile(lower_q), x.quantile(upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


# ---------------------------------------------------------------------------
# Coverage, audit, models, baselines, validation
# ---------------------------------------------------------------------------

def produce_data_coverage(panel: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    layers = [
        ("Climate stress", "CHIRPS", ["rain_season", "rain_annual"]),
        ("Vegetation", "MODIS MOD13Q1", ["ndvi", "evi"]),
        ("Nightlights", "VIIRS Black Marble", ["nightlights"]),
        ("Population", "WorldPop", ["population"]),
        ("Macro context", "World Bank WDI", ["gdp_pc_constant", "ag_value_added_growth", "electricity_access"]),
    ]
    rows = []
    n = len(panel)
    for layer, source, cols in layers:
        present_cols = [c for c in cols if c in panel.columns]
        available = sum(panel[c].notna().sum() for c in present_cols) if present_cols else 0
        possible = n * max(1, len(present_cols)) if present_cols else 0
        rows.append(
            {
                "layer": layer,
                "source": source,
                "variables": ", ".join(present_cols) if present_cols else "not found",
                "rows": n,
                "non_missing_values": int(available),
                "possible_values": int(possible),
                "coverage_rate": available / possible if possible else np.nan,
                "start_year": panel["year"].min(),
                "end_year": panel["year"].max(),
                "spatial_units": panel["district_id"].nunique(),
                "reproducibility_status": "scripted public source" if present_cols else "missing",
            }
        )
    cov = pd.DataFrame(rows)
    save_table(cov, paths.main_tables / "table1_data_sources_coverage.csv", paths.main_tables / "table1_data_sources_coverage.tex")

    # Missingness heatmap data for appendix.
    key_cols = ["rain_season", "ndvi", "evi", "nightlights", "population", "gdp_pc_constant", "electricity_access"]
    key_cols = [c for c in key_cols if c in panel.columns]
    miss = panel.groupby("year")[key_cols].apply(lambda x: x.isna().mean()).reset_index()
    save_table(miss, paths.appendix_tables / "missingness_by_year.csv")
    plot_missingness_heatmap(miss, paths.figures / "fig_data_missingness_heatmap.png")
    return cov


def fit_district_response_models(df: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    cols = ["ndvi_anomaly", "rain_stress", "capacity_index", "exposed_population_share", "district_id", "year"]
    model_df = df[cols].dropna().copy()
    if len(model_df) < 50 or model_df["district_id"].nunique() < 5:
        logging.warning("Too few district observations for fixed-effects model; saving empty model table.")
        res_df = pd.DataFrame([{"model": "district_fe", "status": "insufficient_data", "n": len(model_df)}])
        save_table(res_df, paths.main_tables / "table2_district_response_models.csv", paths.main_tables / "table2_district_response_models.tex")
        return res_df

    formula = "ndvi_anomaly ~ rain_stress + capacity_index + rain_stress:capacity_index + exposed_population_share + C(district_id) + C(year)"
    model = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["district_id"]})
    keep = ["rain_stress", "capacity_index", "rain_stress:capacity_index", "exposed_population_share"]
    rows = []
    for term in keep:
        rows.append(
            {
                "model": "district_FE_NDVI_response",
                "term": term,
                "coef": model.params.get(term, np.nan),
                "std_error": model.bse.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
                "n": int(model.nobs),
                "r2": model.rsquared,
                "interpretation": district_term_interpretation(term),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, paths.main_tables / "table2_district_response_models.csv", paths.main_tables / "table2_district_response_models.tex")
    (paths.appendix_tables / "district_fe_model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    return out


def district_term_interpretation(term: str) -> str:
    return {
        "rain_stress": "Association between rainfall stress and NDVI anomaly, holding district/year effects constant.",
        "capacity_index": "Association between structural capacity proxy and NDVI anomaly.",
        "rain_stress:capacity_index": "Whether capacity buffers or amplifies rainfall stress response.",
        "exposed_population_share": "Association between exposed population share and vegetation condition.",
    }.get(term, "")


def fit_macro_model(wdi: pd.DataFrame, country_remote: Optional[pd.DataFrame], config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    """Estimate simple macro climate-economy model if country-level stats exist."""
    if country_remote is None or country_remote.empty:
        out = pd.DataFrame([{"model": "macro_FE", "status": "skipped_no_country_remote_sensing"}])
        save_table(out, paths.main_tables / "table3_macro_models.csv", paths.main_tables / "table3_macro_models.tex")
        return out
    macro = country_remote.merge(wdi, on=["iso3", "year"], how="left")
    if "ag_value_added_growth" not in macro.columns:
        out = pd.DataFrame([{"model": "macro_FE", "status": "skipped_no_ag_value_added_growth"}])
        save_table(out, paths.main_tables / "table3_macro_models.csv", paths.main_tables / "table3_macro_models.tex")
        return out
    # Construct simple country-level stress/capacity.
    macro["rain_stress"] = macro.groupby("iso3")["rain_season"].transform(lambda x: -((x - x.mean()) / (x.std() + 1e-8)))
    cap_components = []
    if "nightlights" in macro.columns:
        macro["nightlights_norm"] = robust_minmax(macro["nightlights"])
        cap_components.append("nightlights_norm")
    if "electricity_access" in macro.columns:
        macro["electricity_norm"] = robust_minmax(macro["electricity_access"])
        cap_components.append("electricity_norm")
    if cap_components:
        macro["capacity"] = macro[cap_components].mean(axis=1)
    else:
        macro["capacity"] = np.nan
    model_df = macro[["ag_value_added_growth", "rain_stress", "capacity", "iso3", "year", "gdp_pc_constant"]].dropna()
    if len(model_df) < 30 or model_df["iso3"].nunique() < 3:
        out = pd.DataFrame([{"model": "macro_FE", "status": "insufficient_data", "n": len(model_df)}])
        save_table(out, paths.main_tables / "table3_macro_models.csv", paths.main_tables / "table3_macro_models.tex")
        return out
    formula = "ag_value_added_growth ~ rain_stress + capacity + rain_stress:capacity + gdp_pc_constant + C(iso3) + C(year)"
    model = smf.ols(formula, data=model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["iso3"]})
    rows = []
    for term in ["rain_stress", "capacity", "rain_stress:capacity", "gdp_pc_constant"]:
        rows.append(
            {
                "model": "country_FE_agriculture_growth",
                "term": term,
                "coef": model.params.get(term, np.nan),
                "std_error": model.bse.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
                "n": int(model.nobs),
                "r2": model.rsquared,
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, paths.main_tables / "table3_macro_models.csv", paths.main_tables / "table3_macro_models.tex")
    (paths.appendix_tables / "macro_fe_model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    return out


def run_baseline_comparisons(df: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    """Compare integrated priority score with single-layer and ML baselines."""
    top_k = int(config.get("evaluation", {}).get("top_k", 20))
    eval_df = df.copy()
    eval_df["target_adverse_vegetation"] = robust_minmax(eval_df["adverse_vegetation"])

    # Construct transparent ranking scores.
    scores = pd.DataFrame(index=eval_df.index)
    scores["rainfall_only"] = eval_df["S_norm"]
    scores["ndvi_only"] = eval_df["A_bad_norm"]
    scores["weak_capacity_only"] = 1 - eval_df["C_norm"]
    scores["population_exposure_only"] = eval_df["E_norm"]
    scores["equal_weight_index"] = eval_df[["S_norm", "A_bad_norm", "E_norm"]].join(1 - eval_df[["C_norm"]]).mean(axis=1, skipna=True)

    pca_cols = eval_df[["S_norm", "A_bad_norm", "E_norm", "C_norm"]].copy()
    pca_cols["weak_capacity"] = 1 - pca_cols.pop("C_norm")
    pca_ready = pca_cols.fillna(pca_cols.median())
    if len(pca_ready) > 10:
        pca = PCA(n_components=1, random_state=int(config.get("evaluation", {}).get("random_seed", 2026)))
        raw = pca.fit_transform(pca_ready).ravel()
        # Orient PCA so that it positively correlates with adverse vegetation or stress.
        if np.corrcoef(raw, eval_df["target_adverse_vegetation"].fillna(eval_df["target_adverse_vegetation"].median()))[0, 1] < 0:
            raw = -raw
        scores["pca_index"] = robust_minmax(pd.Series(raw, index=eval_df.index))
    else:
        scores["pca_index"] = np.nan
    scores["improved_ras_score"] = eval_df["priority_score"]

    # Predictive ML baselines for adverse vegetation; evaluated with leave-one-year-out.
    ml_metrics, ml_scores = ml_leave_year_out(eval_df, config)
    for name, arr in ml_scores.items():
        scores[name] = robust_minmax(pd.Series(arr, index=eval_df.index))

    target = eval_df["target_adverse_vegetation"]
    rows = []
    true_top = set(target.sort_values(ascending=False).head(top_k).index)
    for method in scores.columns:
        s = scores[method]
        valid = s.notna() & target.notna()
        if valid.sum() < 10:
            rows.append({"method": method, "status": "insufficient_data"})
            continue
        pred = s[valid]
        y = target[valid]
        top = set(pred.sort_values(ascending=False).head(top_k).index)
        rows.append(
            {
                "method": method,
                "status": "ok",
                "rmse_vs_adverse_vegetation": math.sqrt(mean_squared_error(y, pred)),
                "mae_vs_adverse_vegetation": mean_absolute_error(y, pred),
                "spearman_rank_corr": spearman_safe(y, pred),
                "top_k_overlap_with_observed_adverse": len(top & true_top) / max(1, min(top_k, len(true_top))),
                "n": int(valid.sum()),
                "interpretation": baseline_interpretation(method),
            }
        )
    out = pd.DataFrame(rows)
    if ml_metrics:
        save_table(pd.DataFrame(ml_metrics), paths.appendix_tables / "ml_leave_year_out_metrics.csv")
    save_table(out, paths.main_tables / "table4_baseline_comparison.csv", paths.main_tables / "table4_baseline_comparison.tex")
    scores_out = pd.concat([eval_df[["district_id", "district_name", "year", "target_adverse_vegetation"]], scores], axis=1)
    save_table(scores_out, paths.appendix_tables / "all_baseline_scores.csv")
    plot_baseline_comparison(out, paths.figures / "fig_baseline_comparison.png")
    return out


def ml_leave_year_out(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    features = ["rain_stress", "capacity_index", "exposed_population_share", "population", "nightlights", "year"]
    features = [c for c in features if c in df.columns]
    cat_features = ["district_id"] if "district_id" in df.columns else []
    target_col = "target_adverse_vegetation"
    valid = df[features + cat_features + [target_col, "year"]].dropna(subset=[target_col]).copy()
    if len(valid) < 50 or valid["year"].nunique() < 4:
        return [], {"random_forest": np.full(len(df), np.nan), "gradient_boosting": np.full(len(df), np.nan)}

    numeric_features = [c for c in features if c != "year"] + ["year"]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )
    seed = int(config.get("evaluation", {}).get("random_seed", 2026))
    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }
    preds_full = {name: np.full(len(df), np.nan) for name in models}
    metrics: List[Dict[str, Any]] = []
    logo = LeaveOneGroupOut()
    X = valid[numeric_features + cat_features]
    y = valid[target_col]
    groups = valid["year"]
    valid_indices = valid.index.to_numpy()
    for name, model in models.items():
        fold_rows = []
        for train_idx, test_idx in logo.split(X, y, groups=groups):
            pipe = Pipeline([("prep", preprocess), ("model", model)])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            preds_full[name][valid_indices[test_idx]] = pred
            fold_rows.append(
                {
                    "model": name,
                    "held_out_year": int(groups.iloc[test_idx].iloc[0]),
                    "rmse": math.sqrt(mean_squared_error(y.iloc[test_idx], pred)),
                    "mae": mean_absolute_error(y.iloc[test_idx], pred),
                    "r2": r2_score(y.iloc[test_idx], pred) if len(test_idx) > 2 else np.nan,
                    "n_test": len(test_idx),
                }
            )
        metrics.extend(fold_rows)
    return metrics, preds_full


def baseline_interpretation(method: str) -> str:
    return {
        "rainfall_only": "Priority based only on rainfall stress.",
        "ndvi_only": "Priority based only on adverse vegetation condition.",
        "weak_capacity_only": "Priority based only on weak capacity proxy.",
        "population_exposure_only": "Priority based only on exposed population share.",
        "equal_weight_index": "Simple transparent composite benchmark.",
        "pca_index": "Unsupervised data-driven weighting benchmark.",
        "random_forest": "Predictive ML baseline for adverse vegetation; not primary interpretation.",
        "gradient_boosting": "Predictive ML baseline for adverse vegetation; not primary interpretation.",
        "improved_ras_score": "Proposed macro--micro decision score.",
    }.get(method, "")


def spearman_safe(a: pd.Series, b: pd.Series) -> float:
    try:
        val = spearmanr(a, b, nan_policy="omit").correlation
        return float(val) if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def build_top_priority_table(df: pd.DataFrame, config: Dict[str, Any], paths: Paths) -> pd.DataFrame:
    latest_year = int(df["year"].max())
    top_k = int(config.get("evaluation", {}).get("top_k", 20))
    latest = df[df["year"] == latest_year].copy()
    cols = [
        "district_name",
        "year",
        "priority_score",
        "action_category",
        "rain_stress",
        "capacity_index",
        "exposed_population_share",
        "ndvi_anomaly",
        "U_norm",
        "population",
    ]
    cols = [c for c in cols if c in latest.columns]
    top = latest.sort_values("priority_score", ascending=False)[cols].head(top_k)
    top = top.rename(
        columns={
            "rain_stress": "climate_stress",
            "capacity_index": "capacity",
            "U_norm": "uncertainty",
            "ndvi_anomaly": "vegetation_response",
        }
    )
    save_table(top, paths.main_tables / "table5_top_priority_districts.csv", paths.main_tables / "table5_top_priority_districts.tex")
    plot_top_priority(top, paths.figures / "fig_top_priority_districts.png")
    return top


def robustness_summary(df: pd.DataFrame, baseline_table: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    rows = []
    # Ranking stability: year-to-year Spearman among priority scores by district.
    piv = df.pivot_table(index="district_id", columns="year", values="priority_score", aggfunc="mean")
    cors = []
    years = sorted([c for c in piv.columns if isinstance(c, (int, np.integer))])
    for y0, y1 in zip(years[:-1], years[1:]):
        a, b = piv[y0], piv[y1]
        valid = a.notna() & b.notna()
        if valid.sum() > 5:
            cors.append(spearman_safe(a[valid], b[valid]))
    rows.append({"check": "year_to_year_priority_rank_stability", "value": np.nanmean(cors) if cors else np.nan, "interpretation": "Mean adjacent-year Spearman correlation of district priority rankings."})
    rows.append({"check": "missingness_core_layers", "value": df[["rain_stress", "ndvi_anomaly", "capacity_index", "exposed_population_share"]].isna().mean().mean(), "interpretation": "Average missingness across the four core district-year layers."})
    ras_row = baseline_table[baseline_table["method"] == "improved_ras_score"]
    if not ras_row.empty and "spearman_rank_corr" in ras_row:
        rows.append({"check": "ras_rank_corr_with_adverse_vegetation", "value": ras_row["spearman_rank_corr"].iloc[0], "interpretation": "Rank association between proposed score and adverse vegetation outcome."})
    out = pd.DataFrame(rows)
    save_table(out, paths.main_tables / "table6_robustness_summary.csv", paths.main_tables / "table6_robustness_summary.tex")
    return out


# ---------------------------------------------------------------------------
# Transfer test: country-level remote-sensing extraction
# ---------------------------------------------------------------------------

def build_country_remote_sensing(config: Dict[str, Any], paths: Paths, force: bool = False) -> Optional[pd.DataFrame]:
    out = paths.interim / "east_africa_country_year_remote_sensing.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    if not config.get("gee", {}).get("use_gee", True):
        return None
    try:
        ee = init_earth_engine(config)
    except Exception as exc:
        logging.warning("Skipping country transfer GEE extraction: %s", exc)
        return None
    rows = []
    for c in config.get("transfer_countries", []):
        iso3 = c["iso3"]
        try:
            bpath = download_geoboundaries(iso3, "ADM0", paths.raw / "boundaries", None)
            cgdf = gpd.read_file(bpath).to_crs("EPSG:4326")
            cgdf["district_id"] = iso3
            cgdf["district_name"] = c.get("name", iso3)
            fc = gdf_to_ee_feature_collection(cgdf[["district_id", "district_name", "geometry"]], ee)
            stats = extract_country_year_basic(fc, iso3, c.get("name", iso3), config, ee)
            rows.append(stats)
        except Exception as exc:
            logging.warning("Country transfer extraction failed for %s: %s", iso3, exc)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out, index=False)
    return df


def extract_country_year_basic(fc: Any, iso3: str, name: str, config: Dict[str, Any], ee: Any) -> pd.DataFrame:
    gee = config["gee"]["collections"]
    season_months = list(config.get("season_months", [3, 4, 5, 6]))
    records = []
    for year in range(int(config["start_year"]), int(config["end_year"]) + 1):
        season_start = f"{year}-{min(season_months):02d}-01"
        last_month = max(season_months)
        season_end = f"{year+1}-01-01" if last_month == 12 else f"{year}-{last_month+1:02d}-01"
        year_start, year_end = f"{year}-01-01", f"{year+1}-01-01"
        try:
            chirps = ee.ImageCollection(gee["chirps"]["id"]).select(gee["chirps"]["band"])
            rain_img = chirps.filterDate(season_start, season_end).sum().rename("rain_season")
            rain_df = ee_reduce_regions_to_df(rain_img, fc, ee, scale=int(gee["chirps"].get("scale", 5500)), reducer="mean")
            modis_cfg = gee["modis"]
            bands = modis_cfg.get("bands", ["NDVI", "EVI"])
            sf = float(modis_cfg.get("scale_factor", 0.0001))
            def scale_modis(img):
                return img.select(bands).multiply(sf).copyProperties(img, ["system:time_start"])
            modis_img = ee.ImageCollection(modis_cfg["id"]).filterDate(season_start, season_end).map(scale_modis).mean().rename([b.lower() for b in bands])
            veg_df = ee_reduce_regions_to_df(modis_img, fc, ee, scale=int(modis_cfg.get("scale", 250)), reducer="mean")
            rec = {"iso3": iso3, "country": name, "year": year}
            for d in [rain_df, veg_df]:
                for col in d.columns:
                    if col not in {"district_id", "district_name"}:
                        rec[col] = d[col].iloc[0]
            viirs_cfg = gee.get("viirs", {})
            if year >= int(viirs_cfg.get("start_year", 2012)):
                viirs_img = ee.ImageCollection(viirs_cfg["id"]).filterDate(year_start, year_end).select(viirs_cfg["band"]).mean().rename("nightlights")
                viirs_df = ee_reduce_regions_to_df(viirs_img, fc, ee, scale=int(viirs_cfg.get("scale", 500)), reducer="mean")
                rec["nightlights"] = viirs_df.get("nightlights", pd.Series([np.nan])).iloc[0]
            records.append(rec)
        except Exception as exc:
            logging.warning("Country-year extraction failed %s %s: %s", iso3, year, exc)
    return normalize_remote_columns(pd.DataFrame(records))


def transferability_table(country_remote: Optional[pd.DataFrame], paths: Paths) -> pd.DataFrame:
    if country_remote is None or country_remote.empty:
        out = pd.DataFrame([{"status": "skipped_no_country_remote_sensing"}])
        save_table(out, paths.appendix_tables / "east_africa_transferability.csv")
        return out
    cols = ["rain_season", "ndvi", "evi", "nightlights"]
    cols = [c for c in cols if c in country_remote.columns]
    rows = []
    for iso3, g in country_remote.groupby("iso3"):
        row = {"iso3": iso3, "country": g.get("country", pd.Series([iso3])).iloc[0], "years": g["year"].nunique()}
        for col in cols:
            row[f"{col}_coverage"] = g[col].notna().mean()
        row["transferability_score"] = np.nanmean([row[f"{c}_coverage"] for c in cols]) if cols else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("transferability_score", ascending=False)
    save_table(out, paths.appendix_tables / "east_africa_transferability.csv")
    return out


# ---------------------------------------------------------------------------
# Figures and maps
# ---------------------------------------------------------------------------

def plot_missingness_heatmap(miss: pd.DataFrame, outpath: Path) -> None:
    if miss.empty or len(miss.columns) <= 1:
        return
    data = miss.set_index("year")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(data.T.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(data.columns)))
    ax.set_yticklabels(data.columns)
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index.astype(int), rotation=90, fontsize=7)
    ax.set_title("Missingness by year and variable")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Missing fraction")
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_choropleth(gdf: gpd.GeoDataFrame, column: str, title: str, outpath: Path, categorical: bool = False) -> None:
    if column not in gdf.columns:
        logging.warning("Cannot plot missing column %s", column)
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    if categorical:
        cats = ["Act now", "Measure first", "Monitor", "Lower priority"]
        colors = ["#b2182b", "#ef8a62", "#67a9cf", "#d1e5f0"]
        cmap = ListedColormap(colors)
        code = gdf[column].map({c: i for i, c in enumerate(cats)})
        plot_gdf = gdf.copy()
        plot_gdf["_cat_code"] = code
        plot_gdf.plot(column="_cat_code", ax=ax, cmap=cmap, edgecolor="white", linewidth=0.3, missing_kwds={"color": "lightgrey"})
        # Manual legend.
        handles = [plt.Line2D([0], [0], marker="s", color="w", label=c, markerfacecolor=colors[i], markersize=10) for i, c in enumerate(cats)]
        ax.legend(handles=handles, loc="lower left", frameon=True)
    else:
        gdf.plot(column=column, ax=ax, legend=True, edgecolor="white", linewidth=0.3, missing_kwds={"color": "lightgrey", "label": "Missing"})
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def make_maps_and_figures(df: pd.DataFrame, gdf: gpd.GeoDataFrame, paths: Paths) -> None:
    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year].copy()
    map_gdf = gdf.merge(latest, on=["district_id", "district_name"], how="left")
    map_gdf.to_file(paths.maps / f"district_priority_{latest_year}.geojson", driver="GeoJSON")
    plot_choropleth(map_gdf, "rain_stress", f"District rainfall stress, {latest_year}", paths.maps / "fig3_climate_stress_map.png")
    plot_choropleth(map_gdf, "ndvi_anomaly", f"Vegetation response (NDVI anomaly), {latest_year}", paths.maps / "fig4_vegetation_response_map.png")
    plot_ras_surface(df, paths.figures / "fig5_resilience_asymmetry_surface.png")
    plot_choropleth(map_gdf, "priority_score", f"District resilience priority score, {latest_year}", paths.maps / "fig6_priority_map.png")
    plot_choropleth(map_gdf, "action_category", f"Planning typology, {latest_year}", paths.maps / "fig7_action_typology_map.png", categorical=True)
    plot_choropleth(map_gdf, "U_norm", f"Priority uncertainty, {latest_year}", paths.maps / "fig_uncertainty_map.png")
    plot_workflow(paths.figures / "fig1_workflow.png")
    plot_data_architecture(paths.figures / "fig2_data_architecture.png")


def plot_ras_surface(df: pd.DataFrame, outpath: Path) -> None:
    # Surface is descriptive: priority as a function of stress and capacity,
    # holding exposure, adverse vegetation, and uncertainty at their medians.
    stress_grid = np.linspace(0, 1, 80)
    capacity_grid = np.linspace(0, 1, 80)
    SS, CC = np.meshgrid(stress_grid, capacity_grid)
    med_E = float(df["E_norm"].median()) if "E_norm" in df else 0.5
    med_A = float(df["A_bad_norm"].median()) if "A_bad_norm" in df else 0.5
    med_U = float(df["U_norm"].median()) if "U_norm" in df else 0.2
    # Use the same default weights for readability; the exact weights are saved in audit JSON.
    P = 0.30 * SS + 0.22 * (1 - CC) + 0.20 * med_E + 0.20 * med_A + 0.08 * med_U
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cp = ax.contourf(SS, CC, P, levels=25)
    fig.colorbar(cp, ax=ax, label="Priority score")
    ax.set_xlabel("Climate stress (normalized)")
    ax.set_ylabel("Structural capacity (normalized)")
    ax.set_title("Improved macro--micro resilience priority surface")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_baseline_comparison(table: pd.DataFrame, outpath: Path) -> None:
    if table.empty or "spearman_rank_corr" not in table.columns:
        return
    ok = table[table.get("status", "ok") == "ok"].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ok = ok.sort_values("spearman_rank_corr", ascending=True)
    ax.barh(ok["method"], ok["spearman_rank_corr"])
    ax.set_xlabel("Spearman correlation with adverse vegetation ranking")
    ax.set_title("Baseline comparison: ranking agreement with observed adverse vegetation")
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_top_priority(top: pd.DataFrame, outpath: Path) -> None:
    if top.empty or "priority_score" not in top.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    tmp = top.sort_values("priority_score", ascending=True)
    ax.barh(tmp["district_name"], tmp["priority_score"])
    ax.set_xlabel("Priority score")
    ax.set_title("Top priority districts")
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_workflow(outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    boxes = [
        (0.05, 0.65, "Open data\nCHIRPS, MODIS, WDI, WorldPop, VIIRS"),
        (0.30, 0.65, "District-year panel\nfixed boundaries + metadata"),
        (0.55, 0.65, "Models and baselines\nFE, RAS, RF/GB, audits"),
        (0.80, 0.65, "Decision outputs\npriority, uncertainty, action"),
        (0.30, 0.25, "Appendix evidence\nmissingness, sensitivity, validation"),
        (0.60, 0.25, "Main text evidence\ntables, maps, action register"),
    ]
    for x, y, txt in boxes:
        ax.text(x, y, txt, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), fontsize=10)
    for x0, y0, x1, y1 in [(0.15, 0.65, 0.25, 0.65), (0.40, 0.65, 0.50, 0.65), (0.65, 0.65, 0.75, 0.65), (0.55, 0.55, 0.35, 0.35), (0.60, 0.55, 0.60, 0.35)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Open resilience workflow: from public data to district action register", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_data_architecture(outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    layers = ["Rainfall stress\nCHIRPS", "Vegetation\nMODIS NDVI/EVI", "Macro context\nWDI/FAOSTAT", "Exposure\nWorldPop", "Capacity\nVIIRS/WDI"]
    for i, label in enumerate(layers):
        ax.text(0.1 + i * 0.2, 0.72, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"), fontsize=9)
        ax.annotate("", xy=(0.5, 0.42), xytext=(0.1 + i * 0.2, 0.62), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.35, "Harmonized district-year panel", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), fontsize=11)
    ax.annotate("", xy=(0.5, 0.15), xytext=(0.5, 0.27), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.08, "Priority score + action category + uncertainty", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"), fontsize=11)
    ax.set_title("Data architecture for real-data macro--micro resilience workflow")
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str, skip_gee: bool = False, force_gee: bool = False) -> None:
    config = load_config(config_path)
    paths = setup_paths(config)
    setup_logging(paths)
    save_json(config, paths.outputs / "run_config.json")

    logging.info("Starting Open Resilience Uganda pipeline")
    boundary_path = download_geoboundaries(config["country_iso3"], config.get("admin_level", "ADM2"), paths.raw / "boundaries", config.get("boundary_path"))
    gdf = load_boundaries(boundary_path, config)

    # WDI: Uganda + transfer countries.
    transfer_iso3 = sorted({c["iso3"] for c in config.get("transfer_countries", [])} | {config["country_iso3"]})
    wdi = download_wdi(transfer_iso3, config.get("world_bank_indicators", {}), int(config["start_year"]), int(config["end_year"]), paths.raw / "wdi")
    wdi_uganda = wdi[wdi["iso3"] == config["country_iso3"]].copy()

    # Remote sensing district-year extraction.
    remote_path = paths.interim / "district_year_remote_sensing.csv"
    if skip_gee:
        if not remote_path.exists():
            raise FileNotFoundError(
                f"--skip-gee was requested, but {remote_path} does not exist. "
                "Provide a real extracted district-year CSV or run without --skip-gee."
            )
        remote = pd.read_csv(remote_path)
    else:
        remote = build_gee_district_year_stats(gdf, config, paths, force=force_gee)

    panel = build_district_year_panel(remote, gdf, wdi_uganda, config, paths)
    produce_data_coverage(panel, config, paths)
    indexed = add_indices(panel, config, paths)

    # Main empirical outputs.
    fit_district_response_models(indexed, config, paths)
    country_remote = None if skip_gee else build_country_remote_sensing(config, paths, force=force_gee)
    fit_macro_model(wdi, country_remote, config, paths)
    baseline_table = run_baseline_comparisons(indexed, config, paths)
    build_top_priority_table(indexed, config, paths)
    robustness_summary(indexed, baseline_table, paths)
    transferability_table(country_remote, paths)
    make_maps_and_figures(indexed, gdf, paths)

    # Optional validation audit files.
    faostat = load_optional_faostat(config, paths)
    validation_audit = {
        "faostat_supplied": faostat is not None,
        "restricted_microdata_bundled": False,
        "note": "LSMS/UNPS and DHS raw microdata should not be redistributed. Provide access instructions and derived permitted outputs only.",
    }
    save_json(validation_audit, paths.appendix_tables / "external_validation_audit.json")

    write_outputs_readme(paths)
    logging.info("Pipeline finished. Main outputs: %s", paths.outputs)


def write_outputs_readme(paths: Paths) -> None:
    text = """# Open Resilience Uganda outputs

Main-text outputs are in `outputs/main_text/`:

- `tables/table1_data_sources_coverage.csv`
- `tables/table2_district_response_models.csv`
- `tables/table3_macro_models.csv`
- `tables/table4_baseline_comparison.csv`
- `tables/table5_top_priority_districts.csv`
- `tables/table6_robustness_summary.csv`
- `figures/fig1_workflow.png`
- `figures/fig2_data_architecture.png`
- `figures/fig5_resilience_asymmetry_surface.png`
- `maps/fig3_climate_stress_map.png`
- `maps/fig4_vegetation_response_map.png`
- `maps/fig6_priority_map.png`
- `maps/fig7_action_typology_map.png`

Appendix/supplementary outputs are in `outputs/appendix/`:

- missingness and coverage diagnostics
- model summaries
- all baseline scores
- ML leave-one-year-out metrics
- East Africa transferability table
- index construction audit JSON
- external validation audit JSON

Interpretation rule: these outputs are decision-support evidence, not automatic
funding decisions. District categories should be reviewed with local context.
"""
    (paths.outputs / "README_outputs.md").write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the real-data Open Resilience Uganda pipeline.")
    p.add_argument("--config", default="configs/uganda.yaml", help="Path to YAML config.")
    p.add_argument("--skip-gee", action="store_true", help="Use pre-extracted data/interim/district_year_remote_sensing.csv instead of Earth Engine.")
    p.add_argument("--force-gee", action="store_true", help="Ignore cached GEE extraction files and recompute.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config, skip_gee=args.skip_gee, force_gee=args.force_gee)
