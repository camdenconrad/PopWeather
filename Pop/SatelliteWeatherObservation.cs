using System.Globalization;
using System.Text;
using repliKate;

// ============================================================================
// SATELLITE WEATHER SYSTEM - Complete standalone implementation
// Use this for satellite-enhanced weather data
// Your existing WeatherDataParser/WeatherTensorEncoder remain unchanged
// ============================================================================

namespace Pop
{
    // ============================================================================
    // SATELLITE WEATHER OBSERVATION
    // ============================================================================
    public class SatelliteWeatherObservation
    {
        // Base weather data
        public string Date { get; set; }
        public int Hour { get; set; }
        public double Temperature { get; set; }
        public double Humidity { get; set; }
        public double Pressure { get; set; }
        public string Condition { get; set; }
        public double WindSpeed { get; set; }
        public double Precipitation { get; set; }
        public double CloudCover { get; set; }
        public double Visibility { get; set; }

        // Derived features
        public DateTime DateTime { get; set; }
        public int DayOfWeek { get; set; }
        public int TemperatureBucket { get; set; }
        public int HumidityBucket { get; set; }
        public int PressureBucket { get; set; }
        public int ConditionCode { get; set; }
        public double TempChange24h { get; set; }
        public double PressureChange24h { get; set; }

        // Satellite features (30 statistical features)
        public double[] SatelliteFeatures { get; set; }
    }

    // ============================================================================
    // SATELLITE WEATHER PREDICTION
    // ============================================================================
    public class SatelliteWeatherPrediction
    {
        public int TemperatureBucket { get; set; }
        public double PredictedTemperature { get; set; }
        public int HumidityBucket { get; set; }
        public double PredictedHumidity { get; set; }
        public int PressureBucket { get; set; }
        public double PredictedPressure { get; set; }
        public int ConditionCode { get; set; }
        public string PredictedCondition { get; set; }
        public double[] SatelliteFeatures { get; set; }
        public double CloudCoverageEstimate { get; set; }
        public double BrightnessLevel { get; set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append($"{PredictedTemperature:F1}°C, {PredictedHumidity:F0}%, {PredictedPressure:F0}hPa, {PredictedCondition}");
            if (SatelliteFeatures != null)
            {
                sb.Append($", Cloud: {CloudCoverageEstimate:F0}%, Bright: {BrightnessLevel:F0}%");
            }
            return sb.ToString();
        }
    }

    // ============================================================================
    // SATELLITE TENSOR ENCODER (46 dimensions)
    // ============================================================================
    public class SatelliteTensorEncoder
    {
        public const int TENSOR_SIZE = 46;
        public const int WEATHER_FEATURES = 16;
        public const int SATELLITE_FEATURES = 30;

        public Tensor Encode(SatelliteWeatherObservation obs)
        {
            float[] data = new float[TENSOR_SIZE];

            // Weather features [0-15]
            double tempFromBucket = (obs.TemperatureBucket * 5.0) - 20.0;
            data[0] = (float)((tempFromBucket + 20.0) / 65.0);

            double humidityFromBucket = obs.HumidityBucket * 10.0 + 5.0;
            data[1] = (float)(humidityFromBucket / 100.0);

            double pressureFromBucket = 980.0 + (obs.PressureBucket * 5.0) + 2.5;
            data[2] = (float)((pressureFromBucket - 980.0) / 60.0);

            if (obs.ConditionCode >= 0 && obs.ConditionCode < 9)
            {
                data[3 + obs.ConditionCode] = 1.0f;
            }

            data[12] = (float)obs.DayOfWeek / 6.0f;
            data[13] = obs.Hour / 23.0f;
            data[14] = (float)Math.Clamp((obs.TempChange24h + 20.0) / 40.0, 0.0, 1.0);
            data[15] = (float)Math.Clamp((obs.PressureChange24h + 30.0) / 60.0, 0.0, 1.0);

            // Satellite features [16-45]
            if (obs.SatelliteFeatures != null && obs.SatelliteFeatures.Length == SATELLITE_FEATURES)
            {
                for (int i = 0; i < SATELLITE_FEATURES; i++)
                {
                    data[WEATHER_FEATURES + i] = (float)obs.SatelliteFeatures[i];
                }
            }
            else
            {
                for (int i = WEATHER_FEATURES; i < TENSOR_SIZE; i++)
                {
                    data[i] = 0.0f;
                }
            }

            return new Tensor(data);
        }

        public SatelliteWeatherPrediction Decode(Tensor tensor)
        {
            var prediction = new SatelliteWeatherPrediction();

            // Decode weather features
            prediction.PredictedTemperature = (tensor.Data[0] * 65.0) - 20.0;
            prediction.TemperatureBucket = (int)Math.Round((prediction.PredictedTemperature + 20.0) / 5.0);

            prediction.PredictedHumidity = tensor.Data[1] * 100.0;
            prediction.HumidityBucket = (int)Math.Round(prediction.PredictedHumidity / 10.0);

            prediction.PredictedPressure = (tensor.Data[2] * 60.0) + 980.0;
            prediction.PressureBucket = (int)Math.Round((prediction.PredictedPressure - 980.0) / 5.0);

            int conditionCode = FindMaxIndex(tensor.Data, 3, 9);
            prediction.ConditionCode = conditionCode;
            prediction.PredictedCondition = ConditionCodeToString(conditionCode);

            // Decode satellite features
            prediction.SatelliteFeatures = new double[SATELLITE_FEATURES];
            for (int i = 0; i < SATELLITE_FEATURES; i++)
            {
                prediction.SatelliteFeatures[i] = tensor.Data[WEATHER_FEATURES + i];
            }

            prediction.CloudCoverageEstimate = AnalyzeCloudCoverage(prediction.SatelliteFeatures);
            prediction.BrightnessLevel = AnalyzeBrightness(prediction.SatelliteFeatures);

            return prediction;
        }

        private double AnalyzeCloudCoverage(double[] features)
        {
            if (features == null || features.Length < SATELLITE_FEATURES) return 0.0;
            double brightPixels = (features[7] + features[16] + features[25]) / 3.0;
            double darkPixels = (features[8] + features[17] + features[26]) / 3.0;
            return Math.Clamp((brightPixels - darkPixels + 1.0) * 50.0, 0.0, 100.0);
        }

        private double AnalyzeBrightness(double[] features)
        {
            if (features == null || features.Length < SATELLITE_FEATURES) return 0.0;
            double redMean = features[0];
            double greenMean = features[9];
            double blueMean = features[18];
            return (redMean + greenMean + blueMean) / 3.0 * 100.0;
        }

        private int FindMaxIndex(float[] data, int start, int length)
        {
            int maxIndex = 0;
            float maxValue = data[start];
            for (int i = 1; i < length; i++)
            {
                if (data[start + i] > maxValue)
                {
                    maxValue = data[start + i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private string ConditionCodeToString(int code)
        {
            return code switch
            {
                0 => "Clear",
                1 => "Partly Cloudy",
                2 => "Cloudy",
                3 => "Rain",
                4 => "Heavy Rain",
                5 => "Snow",
                6 => "Heavy Snow",
                7 => "Fog",
                8 => "Windy",
                _ => "Unknown"
            };
        }
    }

    // ============================================================================
    // SATELLITE DATA PARSER
    // ============================================================================
    public class SatelliteDataParser
    {
        public static List<SatelliteWeatherObservation> ParseCSV(string filePath)
        {
            var observations = new List<SatelliteWeatherObservation>();
            int lineNumber = 0;
            int errorCount = 0;
            bool hasSatelliteData = false;

            using (var reader = new StreamReader(filePath))
            {
                string header = reader.ReadLine();
                lineNumber++;

                if (header == null)
                {
                    Console.WriteLine("❌ Empty file");
                    return observations;
                }

                var headerCols = header.Split(',');
                hasSatelliteData = headerCols.Any(h => h.StartsWith("SatFeat_"));

                if (hasSatelliteData)
                {
                    Console.WriteLine("🛰️  Detected satellite imagery features in CSV");
                }
                else
                {
                    Console.WriteLine("📊 Standard weather data (no satellite features)");
                }

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    lineNumber++;

                    try
                    {
                        var parts = line.Split(',');
                        if (parts.Length < 10) continue;

                        var obs = new SatelliteWeatherObservation
                        {
                            Date = parts[0],
                            Hour = int.Parse(parts[1]),
                            Temperature = double.Parse(parts[2], CultureInfo.InvariantCulture),
                            Humidity = double.Parse(parts[3], CultureInfo.InvariantCulture),
                            Pressure = double.Parse(parts[4], CultureInfo.InvariantCulture),
                            Condition = parts[5],
                            WindSpeed = double.Parse(parts[6], CultureInfo.InvariantCulture),
                            Precipitation = double.Parse(parts[7], CultureInfo.InvariantCulture),
                            CloudCover = parts.Length > 8 && !string.IsNullOrEmpty(parts[8]) ? double.Parse(parts[8], CultureInfo.InvariantCulture) : 0,
                            Visibility = parts.Length > 9 && !string.IsNullOrEmpty(parts[9]) ? double.Parse(parts[9], CultureInfo.InvariantCulture) : 10000
                        };

                        // Parse satellite features if present
                        if (hasSatelliteData && parts.Length >= 40)
                        {
                            obs.SatelliteFeatures = new double[30];
                            for (int i = 0; i < 30; i++)
                            {
                                if (!string.IsNullOrEmpty(parts[10 + i]))
                                {
                                    obs.SatelliteFeatures[i] = double.Parse(parts[10 + i], CultureInfo.InvariantCulture);
                                }
                                else
                                {
                                    obs.SatelliteFeatures[i] = 0.0;
                                }
                            }
                        }

                        // Parse date
                        if (DateTime.TryParse(obs.Date, out DateTime dt))
                        {
                            obs.DateTime = dt.AddHours(obs.Hour);
                            obs.DayOfWeek = (int)obs.DateTime.DayOfWeek;
                        }

                        // Calculate buckets
                        obs.TemperatureBucket = (int)Math.Round((obs.Temperature + 20.0) / 5.0);
                        obs.HumidityBucket = (int)Math.Round(obs.Humidity / 10.0);
                        obs.PressureBucket = (int)Math.Round((obs.Pressure - 980.0) / 5.0);
                        obs.ConditionCode = ConditionToCode(obs.Condition);

                        observations.Add(obs);
                    }
                    catch
                    {
                        errorCount++;
                    }
                }
            }

            Console.WriteLine($"✅ Parsed {observations.Count} observations, {errorCount} errors");
            return observations;
        }

        public static void CalculateHistoricalFeatures(List<SatelliteWeatherObservation> observations)
        {
            Console.WriteLine("🔍 Calculating weather features...");

            for (int i = 0; i < observations.Count; i++)
            {
                var current = observations[i];

                if (i >= 24)
                {
                    var past24h = observations[i - 24];
                    current.TempChange24h = current.Temperature - past24h.Temperature;
                    current.PressureChange24h = current.Pressure - past24h.Pressure;
                }
                else
                {
                    current.TempChange24h = 0.0;
                    current.PressureChange24h = 0.0;
                }
            }

            Console.WriteLine("✅ Weather features calculated");
        }

        public static void PrintDatasetStatistics(List<SatelliteWeatherObservation> observations)
        {
            if (observations.Count == 0) return;

            Console.WriteLine();
            Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║               SATELLITE WEATHER DATASET STATISTICS                     ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
            Console.WriteLine();
            Console.WriteLine($"📊 Total Observations: {observations.Count}");
            Console.WriteLine($"📅 Date Range: {observations.First().Date} to {observations.Last().Date}");
            Console.WriteLine();

            var temps = observations.Select(o => o.Temperature).ToList();
            Console.WriteLine("🌡️  TEMPERATURE:");
            Console.WriteLine($"  Min: {temps.Min():F1}°C, Max: {temps.Max():F1}°C, Avg: {temps.Average():F1}°C");
            Console.WriteLine();

            var conditions = observations.GroupBy(o => o.Condition)
                .OrderByDescending(g => g.Count())
                .ToList();

            Console.WriteLine("🌤️  WEATHER CONDITIONS:");
            foreach (var condition in conditions.Take(8))
            {
                double pct = (condition.Count() * 100.0) / observations.Count;
                string bar = new string('█', (int)(pct / 2));
                Console.WriteLine($"  {condition.Key,-15}: {condition.Count(),6} times ({pct,5:F1}%) {bar}");
            }
            Console.WriteLine();

            var obsWithSatellite = observations.Where(o => o.SatelliteFeatures != null).ToList();
            if (obsWithSatellite.Count > 0)
            {
                Console.WriteLine("🛰️  SATELLITE IMAGERY DATA:");
                Console.WriteLine($"  Observations with satellite: {obsWithSatellite.Count} ({obsWithSatellite.Count * 100.0 / observations.Count:F1}%)");
                Console.WriteLine($"  Feature dimensions: 30");
                Console.WriteLine();
            }

            Console.WriteLine("📋 SAMPLE OBSERVATIONS:");
            foreach (var obs in observations.Take(3))
            {
                string satInfo = obs.SatelliteFeatures != null ? " [+Satellite]" : "";
                Console.WriteLine($"  {obs.DateTime:yyyy-MM-dd HH:mm}: {obs.Temperature:F1}°C, {obs.Humidity:F0}%, {obs.Condition}{satInfo}");
            }
            Console.WriteLine();
        }

        private static int ConditionToCode(string condition)
        {
            return condition switch
            {
                "Clear" => 0,
                "Partly Cloudy" => 1,
                "Cloudy" => 2,
                "Rain" => 3,
                "Heavy Rain" => 4,
                "Snow" => 5,
                "Heavy Snow" => 6,
                "Fog" => 7,
                "Freezing Drizzle" => 7,
                "Freezing Rain" => 7,
                "Drizzle" => 3,
                "Rain Showers" => 3,
                "Snow Showers" => 5,
                "Snow Grains" => 5,
                "Thunderstorm" => 8,
                "Windy" => 8,
                _ => 0
            };
        }
    }
}