#!/usr/bin/env python3
"""
Weather Data Fetcher
Retrieves historical weather data and saves to CSV format for prediction system

Usage:
    python fetch_weather_data.py [--city CITY] [--days DAYS] [--output FILE]

Requirements:
    pip install requests python-dateutil
"""

import requests
import json
import csv
from datetime import datetime, timedelta
import argparse
import sys

# Using Open-Meteo API (free, no API key required)
API_BASE = "https://archive-api.open-meteo.com/v1/archive"

def fetch_weather_data(latitude, longitude, start_date, end_date, output_file="weather_data.csv"):
    """
    Fetch historical weather data from Open-Meteo API

    Args:
        latitude: Latitude of location
        longitude: Longitude of location
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Output CSV filename
    """
    print(f"🌍 Fetching weather data for coordinates ({latitude}, {longitude})")
    print(f"📅 Date range: {start_date} to {end_date}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "weather_code",
            "wind_speed_10m",
            "precipitation"
        ],
        "timezone": "auto"
    }

    try:
        print("📡 Making API request...")
        response = requests.get(API_BASE, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "hourly" not in data:
            print("❌ Error: No hourly data in response")
            return False

        hourly = data["hourly"]
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        humidity = hourly["relative_humidity_2m"]
        pressure = hourly["surface_pressure"]
        weather_codes = hourly["weather_code"]
        wind_speed = hourly["wind_speed_10m"]
        precipitation = hourly["precipitation"]

        print(f"✅ Received {len(times)} hourly observations")
        print(f"📝 Writing to {output_file}...")

        # Write to CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Date", "Hour", "Temperature", "Humidity", 
                "Pressure", "Condition", "WindSpeed", "Precipitation"
            ])

            for i in range(len(times)):
                dt = datetime.fromisoformat(times[i])
                date_str = dt.strftime("%Y-%m-%d")
                hour = dt.hour

                # Convert weather code to condition string
                condition = weather_code_to_condition(weather_codes[i])

                writer.writerow([
                    date_str,
                    hour,
                    round(temps[i], 1),
                    round(humidity[i], 1),
                    round(pressure[i], 1),
                    condition,
                    round(wind_speed[i], 1),
                    round(precipitation[i], 2)
                ])

        print(f"✅ Successfully saved {len(times)} observations to {output_file}")
        print(f"📊 Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C")
        print(f"💧 Avg humidity: {sum(humidity)/len(humidity):.1f}%")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def weather_code_to_condition(code):
    """
    Convert WMO weather code to readable condition
    https://open-meteo.com/en/docs
    """
    code = int(code) if code is not None else 0

    if code == 0:
        return "Clear"
    elif code in [1, 2]:
        return "Partly Cloudy"
    elif code == 3:
        return "Cloudy"
    elif code in [45, 48]:
        return "Fog"
    elif code in [51, 53, 55]:
        return "Drizzle"
    elif code in [56, 57]:
        return "Freezing Drizzle"
    elif code in [61, 63]:
        return "Rain"
    elif code == 65:
        return "Heavy Rain"
    elif code in [66, 67]:
        return "Freezing Rain"
    elif code in [71, 73]:
        return "Snow"
    elif code == 75:
        return "Heavy Snow"
    elif code == 77:
        return "Snow Grains"
    elif code in [80, 81, 82]:
        return "Rain Showers"
    elif code in [85, 86]:
        return "Snow Showers"
    elif code in [95, 96, 99]:
        return "Thunderstorm"
    else:
        return "Clear"

# Predefined locations
LOCATIONS = {
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "sydney": (-33.8688, 151.2093),
    "berlin": (52.5200, 13.4050),
    "toronto": (43.6532, -79.3832),
    "los_angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "seattle": (47.6062, -122.3321),
}

def main():
    parser = argparse.ArgumentParser(description="Fetch historical weather data for prediction")
    parser.add_argument("--city", type=str, default="new_york", 
                       help=f"City name. Options: {', '.join(LOCATIONS.keys())}")
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days of historical data (default: 365)")
    parser.add_argument("--output", type=str, default="weather_data.csv",
                       help="Output CSV file (default: weather_data.csv)")
    parser.add_argument("--lat", type=float, help="Custom latitude")
    parser.add_argument("--lon", type=float, help="Custom longitude")

    args = parser.parse_args()

    # Determine coordinates
    if args.lat is not None and args.lon is not None:
        latitude = args.lat
        longitude = args.lon
        print(f"📍 Using custom coordinates: ({latitude}, {longitude})")
    elif args.city.lower() in LOCATIONS:
        latitude, longitude = LOCATIONS[args.city.lower()]
        print(f"📍 Using predefined location: {args.city.title()}")
    else:
        print(f"❌ Unknown city: {args.city}")
        print(f"Available cities: {', '.join(LOCATIONS.keys())}")
        print(f"Or use --lat and --lon for custom coordinates")
        sys.exit(1)

    # Calculate date range
    end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=args.days)

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          WEATHER DATA FETCHER FOR PREDICTION SYSTEM            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    success = fetch_weather_data(
        latitude, 
        longitude, 
        start_date.isoformat(), 
        end_date.isoformat(),
        args.output
    )

    if success:
        print()
        print("✅ Data fetch complete!")
        print(f"💡 Now run the prediction system:")
        print(f"   dotnet run --project Pop/Pop.csproj {args.output}")
    else:
        print()
        print("❌ Data fetch failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
