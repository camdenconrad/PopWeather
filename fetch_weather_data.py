#!/usr/bin/env python3
"""
Weather Data Fetcher using Real Reanalysis Datasets
Retrieves ML-ready weather data from ERA5, WeatherBench 2, or NASA POWER

Features:
- ERA5 reanalysis data (highest quality, 0.25° resolution)
- WeatherBench 2 dataset (ML-optimized, pre-processed)
- NASA POWER (satellite-derived, lightweight)
- NetCDF format with efficient compression
- Multiple atmospheric levels and variables

Usage:
    # ERA5 data (requires CDS API key):
    python fetch_weather_data.py --source era5 --days 365 --region 30,50,-10,10

    # WeatherBench 2 (pre-downloaded):
    python fetch_weather_data.py --source weatherbench --days 365

    # NASA POWER (lightweight):
    python fetch_weather_data.py --source nasa-power --days 365 --lat 40.7 --lon -74.0
"""

import requests
import csv
from datetime import datetime, timedelta
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import time

API_BASE = "https://archive-api.open-meteo.com/v1/archive"
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
WEATHERBENCH_BASE = "gs://weatherbench2/datasets"
