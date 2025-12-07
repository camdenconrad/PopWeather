# Weather Prediction System Using Tensor Sequence Learning
# Weather Data Fetcher - Real Reanalysis Datasets

Fetch ML-ready weather data from real atmospheric reanalysis sources instead of synthetic satellite imagery.

## 🌟 Features

- **ERA5 Reanalysis**: Highest quality, 0.25° resolution, multiple pressure levels
- **NASA POWER**: Satellite-derived, lightweight, fast downloads
- **WeatherBench 2**: ML-optimized, preprocessed for deep learning

## 📦 Dataset Sizes

| Source | Size per Year | Resolution | Variables |
|--------|--------------|------------|-----------|
| NASA POWER | 50-200 MB | Point data | 7+ variables |
| ERA5 (regional) | 200-900 MB | 0.25° (~25km) | 15+ variables, multi-level |
| WeatherBench | 300 MB - 5 GB | 1.4° to 0.25° | 20+ variables |

**No more GOES-16 multi-TB downloads!**

## 🚀 Quick Start

### 1. Install Dependencies
A machine learning system that uses tensor-based sequence learning to predict weather conditions from historical data. The system employs rolling window validation to test prediction accuracy across different time periods.

## 🌟 Features

- **Real Weather Data**: Fetch historical weather data from free APIs
- **Tensor-Based Learning**: Encodes weather observations into 64-dimensional tensors
- **Multiple Prediction Methods**: Tests 6 different prediction algorithms
- **Rolling Window Validation**: Evaluates performance across different time periods
- **Parallel Processing**: Utilizes all CPU cores for fast experimentation
- **Comprehensive Metrics**: Temperature MAE, condition accuracy, and more

## 📊 What It Predicts

- **Temperature** (bucketed into 5°C ranges)
- **Humidity** (10% buckets)
- **Atmospheric Pressure** (5 hPa buckets)
- **Weather Conditions** (Clear, Cloudy, Rain, Snow, etc.)

## 🚀 Quick Start

### Prerequisites

- .NET 9.0 SDK
- Python 3.7+ (for data fetching)
- Internet connection (to fetch weather data)

### Installation

1. Clone the repository:
