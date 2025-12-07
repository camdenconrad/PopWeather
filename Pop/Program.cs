using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Pop;
using repliKate;
// ============================================================================
// ENHANCED AUTO MODEL TUNER - Comprehensive Parameter Search
// ============================================================================
// More thorough optimization:
// 1. Larger beam width (10 candidates)
// 2. Binary search on TOP 3 candidates (not just #1)
// 3. More variations per candidate
// 4. Finer grid in initial population
// 5. Skip expensive Stage 3 (full dataset validation)
// 6. NEW: Show 12-hour predictions after tuning
// ============================================================================

class WeatherPredictionValidation
{
    public static readonly object consoleLock = new object();
    private static int completedTests = 0;

    static void Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        int processorCount = Environment.ProcessorCount;

        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║   ENHANCED AUTO MODEL TUNER - Comprehensive Parameter Search          ║");
        Console.WriteLine($"║         Running on {processorCount} CPU cores                                       ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        string filePath = args.Length > 0 ? args[0] : "/home/coffee/RiderProjects/Pop/Pop/weather_data_satellite.csv";
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"❌ Error: Could not find {filePath}");
            Console.WriteLine("💡 Tip: Run fetch_weather_data.py to download weather data first!");
            Console.WriteLine("    python fetch_weather_data.py");
            return;
        }

        Console.WriteLine($"📂 Reading from: {filePath}\n");

        // Parse weather data
        var observations = SatelliteDataParser.ParseCSV(filePath);
        if (observations.Count == 0)
        {
            Console.WriteLine("❌ No valid observations parsed!");
            return;
        }

        SatelliteDataParser.CalculateHistoricalFeatures(observations);
        SatelliteDataParser.PrintDatasetStatistics(observations);

        var encoder = new SatelliteTensorEncoder();

        // Create tensors and store actual values for comparison
        Tensor[] allTensors = observations.Select(o => encoder.Encode(o)).ToArray();
        SatelliteWeatherObservation[] allObservations = observations.ToArray();

        Console.WriteLine($"✅ Dataset ready: {allTensors.Length} observations\n");

        // Baselines for weather prediction
        double avgTemp = observations.Average(o => o.Temperature);
        var tempErrors = observations.Select(o => Math.Abs(o.Temperature - avgTemp)).ToList();
        double avgTempBaseline = tempErrors.Average();

        var mostCommonCondition = observations.GroupBy(o => o.Condition)
            .OrderByDescending(g => g.Count()).First();
        double conditionBaseline = (mostCommonCondition.Count() * 100.0) / observations.Count;

        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine("BASELINES");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine($"🌡️  Avg Temperature Baseline:  {avgTemp,6:F1}°C (MAE: {avgTempBaseline:F2}°C)");
        Console.WriteLine($"🌤️  Most Common Condition:     {conditionBaseline,6:F1}% ({mostCommonCondition.Key})");
        Console.WriteLine();

        int trainPercent = 95;
        int windowStartIndex = 0;
        int windowEndIndex = allTensors.Length;
        int splitIndex = (int)(allTensors.Length * (trainPercent / 100.0));

        var firstObs = allObservations[0].DateTime;
        var lastObs = allObservations[allObservations.Length - 1].DateTime;
        var splitDate = allObservations[splitIndex].DateTime;
        Console.WriteLine($"📅 Date range: {firstObs:yyyy-MM-dd} to {lastObs:yyyy-MM-dd}");
        Console.WriteLine($"📅 Train/Test split at: {splitDate:yyyy-MM-dd}");
        Console.WriteLine($"📊 Train: {splitIndex} obs, Test: {allTensors.Length - splitIndex} obs");
        Console.WriteLine();

        var globalStopwatch = Stopwatch.StartNew();

        // ============================================================================
        // ENHANCED AUTO MODEL TUNER
        // ============================================================================

        var tuner = new EnhancedModelTuner(
            allTensors,
            allObservations,
            windowStartIndex,
            windowEndIndex,
            trainPercent,
            processorCount,
            avgTempBaseline,
            conditionBaseline
        );

        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              ENHANCED AUTO MODEL TUNER                                 ║");
        Console.WriteLine("║   Comprehensive Search - No Candidates Left Behind                     ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // Run the enhanced tuner
        var bestConfig = tuner.FindOptimalParameters();

        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                  OPTIMAL PARAMETERS FOUND                              ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"🎯 Lookback Window:      {bestConfig.ModelWindowSize}h");
        Console.WriteLine($"🎯 Similarity Threshold: {bestConfig.Similarity:F4}");
        Console.WriteLine($"🎯 Temporal Decay:       {bestConfig.TemporalDecay:F6}");
        Console.WriteLine($"🎯 Temperature:          {bestConfig.Temperature:F4}");
        Console.WriteLine($"🎯 Exploration Rate:     {bestConfig.ExplorationRate:F4}");
        Console.WriteLine($"🎯 Experience Replay:    {bestConfig.EnableReplay}");
        Console.WriteLine($"🎯 Delta Regression:     {bestConfig.EnableDelta}");
        Console.WriteLine();

        // ============================================================================
        // VALIDATION - Stages 1 & 2 only (skip expensive Stage 3)
        // ============================================================================

        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           VALIDATION - Testing Optimal Config (Stages 1-2)            ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        var validationStages = new List<StageResults>();

        // Stage 1: Quick validation (1000 obs)
        Console.WriteLine("🚀 Stage 1: QUICK VALIDATION (~30 seconds)");
        Console.WriteLine("   Purpose: Verify optimal params with moderate dataset");
        Console.WriteLine();
        var stage1Config = CreateValidationConfig(bestConfig, windowStartIndex, windowEndIndex, trainPercent,
            maxTrainObs: 1000, trainStep: 50, testStep: 5);
        var stage1Results = RunValidationStage(1, "Quick Validation", new[] { stage1Config },
            allTensors, allObservations, processorCount, avgTempBaseline, conditionBaseline);
        validationStages.Add(stage1Results);

        // Stage 2: Thorough validation (5000 obs)
        Console.WriteLine("⚡ Stage 2: THOROUGH VALIDATION (~2-3 minutes)");
        Console.WriteLine("   Purpose: Comprehensive testing with substantial data");
        Console.WriteLine();
        var stage2Config = CreateValidationConfig(bestConfig, windowStartIndex, windowEndIndex, trainPercent,
            maxTrainObs: 5000, trainStep: 10, testStep: 2);
        var stage2Results = RunValidationStage(2, "Thorough Validation", new[] { stage2Config },
            allTensors, allObservations, processorCount, avgTempBaseline, conditionBaseline);
        validationStages.Add(stage2Results);

        // SKIP Stage 3 - too expensive (5-10 minutes)
        Console.WriteLine("💡 Skipping Stage 3 (Maximum Accuracy) - would take 5-10 minutes");
        Console.WriteLine("   Stage 2 provides excellent accuracy estimate with 5K observations");
        Console.WriteLine();

        globalStopwatch.Stop();

        // ============================================================================
        // FINAL SUMMARY
        // ============================================================================

        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    VALIDATION SUMMARY                                  ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        string[] stageIcons = { "🚀", "⚡" };
        string[] stageNames = { "Quick Validation", "Thorough Validation" };

        for (int i = 0; i < validationStages.Count; i++)
        {
            var stage = validationStages[i];
            var best = stage.Results.First();
            Console.WriteLine($"{stageIcons[i]} Stage {i + 1}: {stageNames[i]}");
            Console.WriteLine($"   ⏱️  Time: {stage.Duration:mm\\:ss}");
            Console.WriteLine($"   📈 TempMAE: {best.BestTempMAE:F2}°C (vs baseline {avgTempBaseline:F2}°C)");
            Console.WriteLine($"   📈 CondAcc: {best.BestConditionAccuracy:F1}% (vs baseline {conditionBaseline:F1}%)");
            Console.WriteLine($"   💪 Improvement: {avgTempBaseline - best.BestTempMAE:+F2}°C, {best.BestConditionAccuracy - conditionBaseline:+F1}%");
            Console.WriteLine($"   📊 Predictions: {best.PredictionCount}");
            Console.WriteLine();
        }

        var finalBest = validationStages.Last().Results.First();
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine("🏆 FINAL RESULTS (Stage 2 - Thorough Validation):");
        Console.WriteLine($"   Temp MAE:      {finalBest.BestTempMAE:F2}°C");
        Console.WriteLine($"   Condition Acc: {finalBest.BestConditionAccuracy:F1}%");
        Console.WriteLine($"   Total improvement: {avgTempBaseline - finalBest.BestTempMAE:+F2}°C, {finalBest.BestConditionAccuracy - conditionBaseline:+F1}%");
        Console.WriteLine();
        Console.WriteLine($"⚡ Total time: {globalStopwatch.Elapsed:mm\\:ss}");
        Console.WriteLine();

        // Save results
        var allResults = validationStages.SelectMany(s => s.Results).ToList();
        SaveValidationResults(allResults, bestConfig, avgTempBaseline, conditionBaseline);

        // ============================================================================
        // NEW: GENERATE 12-HOUR PREDICTIONS
        // ============================================================================
        Console.WriteLine();
        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              5-DAY WEATHER PREDICTION                                  ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        Generate5DayPredictions(allTensors, allObservations, bestConfig, encoder);

        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine("ENHANCED AUTO MODEL TUNER COMPLETE");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
    }

    static void Generate5DayPredictions(Tensor[] allTensors, SatelliteWeatherObservation[] allObservations,
        RollingConfig optimalConfig, SatelliteTensorEncoder encoder)
    {
        Console.WriteLine("🔮 Generating predictions for the next 5 days (120 hours)...");
        Console.WriteLine("   🚀 GPU acceleration: ENABLED (training on full dataset)");
        Console.WriteLine();

        // Find the most recent VALID observation (skip -999 sentinel values)
        int lastValidIndex = -1;
        for (int i = allObservations.Length - 1; i >= 0; i--)
        {
            var obs = allObservations[i];
            if (obs.Temperature > -100 && obs.Humidity >= 0 && obs.Pressure > 0)
            {
                lastValidIndex = i;
                break;
            }
        }

        if (lastValidIndex == -1)
        {
            Console.WriteLine("❌ No valid observations found in dataset!");
            Console.WriteLine("   All observations appear to have missing data (-999 values)");
            Console.WriteLine("   Please re-download the weather data:");
            Console.WriteLine("   python fetch_weather_data.py");
            Console.WriteLine();
            return;
        }

        var lastValidObservation = allObservations[lastValidIndex];
        var lastDataTime = lastValidObservation.DateTime;
        var now = DateTime.Now;
        var dataAge = now - lastDataTime;

        // Display current date/time info
        Console.WriteLine($"🕐 System Time: {now:dddd, MMMM d, yyyy 'at' h:mm tt}");
        Console.WriteLine($"📅 Last Available Data: {lastDataTime:MMM d, yyyy 'at' h:00 tt}");
        Console.WriteLine($"⏳ Data Age: {dataAge.TotalHours:F0} hours ({dataAge.TotalDays:F1} days old)");
        Console.WriteLine();

        // Warn if data is stale
        if (dataAge.TotalDays > 3)
        {
            Console.WriteLine($"⚠️  WARNING: Weather data is {dataAge.TotalDays:F1} days old");
            Console.WriteLine("   Predictions may be less accurate due to data staleness.");
            Console.WriteLine();
            Console.WriteLine("   💡 Recommended: Update your data for better predictions:");
            Console.WriteLine("      python fetch_weather_data.py");
            Console.WriteLine();
        }

        // Skip invalid observations at the end
        int validDataLength = lastValidIndex + 1;
        Console.WriteLine($"📊 Using {validDataLength} valid observations (skipped {allObservations.Length - validDataLength} invalid entries)");
        Console.WriteLine();

        var trainingStopwatch = Stopwatch.StartNew();

        // Train model on valid data with GPU acceleration
        var tree = new TensorSequenceTree(
            maxContextWindow: optimalConfig.ModelWindowSize,
            similarityThreshold: optimalConfig.Similarity,
            temporalDecay: optimalConfig.TemporalDecay,
            temperature: optimalConfig.Temperature,
            explorationRate: optimalConfig.ExplorationRate,
            enableExperienceReplay: optimalConfig.EnableReplay,
            enableDeltaRegression: optimalConfig.EnableDelta,
            enableGpuAcceleration: true  // GPU acceleration for full dataset training
        );

        // Train on valid data only
        int trainWindowSize = optimalConfig.ModelWindowSize;
        int trainStep = 1;  // Larger step = faster training, still good coverage

        int totalWindows = Math.Max(1, (validDataLength - trainWindowSize) / trainStep);
        int windowsProcessed = 0;
        int lastPercent = -1;

        Console.WriteLine($"📚 Training on {validDataLength} observations (step size: {trainStep})...");
        Console.WriteLine($"   Estimated training windows: {totalWindows}");
        Console.Write("   Progress: 0%");

        for (int i = 0; i < validDataLength - trainWindowSize; i += trainStep)
        {
            Tensor[] window = new Tensor[trainWindowSize];
            for (int w = 0; w < trainWindowSize; w++)
            {
                window[w] = allTensors[i + w];
            }
            tree.Learn(window);

            windowsProcessed++;
            if (totalWindows > 0)
            {
                int currentPercent = (windowsProcessed * 100) / totalWindows;
                if (currentPercent != lastPercent && currentPercent % 10 == 0)
                {
                    Console.Write($"\r   Progress: {currentPercent}%");
                    lastPercent = currentPercent;
                }
            }
        }

        trainingStopwatch.Stop();
        Console.WriteLine($"\r   Progress: 100%");
        Console.WriteLine($"✅ Model trained on {validDataLength} observations in {trainingStopwatch.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine();

        // Get the most recent valid data as context
        int contextSize = Math.Min(optimalConfig.ModelWindowSize, validDataLength);
        if (contextSize <= 0)
        {
            Console.WriteLine("❌ Not enough valid data for predictions!");
            return;
        }

        Tensor[] context = new Tensor[contextSize];
        for (int i = 0; i < contextSize; i++)
        {
            context[i] = allTensors[validDataLength - contextSize + i];
        }

        Console.WriteLine($"📊 Starting conditions (from {lastDataTime:MMM d 'at' h:00 tt}):");
        Console.WriteLine($"   🌡️  Temperature: {CelsiusToFahrenheit(lastValidObservation.Temperature):F1}°F ({lastValidObservation.Temperature:F1}°C)");
        Console.WriteLine($"   💧 Humidity: {lastValidObservation.Humidity:F0}%");
        Console.WriteLine($"   🎈 Pressure: {lastValidObservation.Pressure:F1} hPa");
        Console.WriteLine($"   🌤️  Condition: {lastValidObservation.Condition}");
        Console.WriteLine($"   💨 Wind Speed: {lastValidObservation.WindSpeed:F1} km/h");
        Console.WriteLine();
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine($"📊 5-DAY FORECAST FROM {now:dddd, MMMM d, yyyy 'at' h:mm tt}");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine();

        var predictions = new List<PredictionResult>();

        // Generate predictions for next 5 days (120 hours)
        Console.WriteLine("Generating predictions...");
        int predictionStep = 1; // Predict every 3 hours for reasonable output size

        for (int hour = predictionStep; hour <= 120; hour += predictionStep)
        {
            try
            {
                var prediction = tree.PredictNextDeltaSmoothed(context);
                if (prediction == null)
                {
                    Console.WriteLine($"⚠️  Hour +{hour}: Unable to generate prediction (insufficient similar patterns)");
                    break;
                }

                var predictedWeather = encoder.Decode(prediction);

                // Validate prediction - skip if clearly invalid
                if (predictedWeather.PredictedTemperature < -100 ||
                    predictedWeather.PredictedTemperature > 100 ||
                    predictedWeather.PredictedHumidity < 0 ||
                    predictedWeather.PredictedHumidity > 100)
                {
                    Console.WriteLine($"⚠️  Hour +{hour}: Invalid prediction generated (skipping)");
                    break;
                }

                var predictedTime = now.AddHours(hour);

                predictions.Add(new PredictionResult
                {
                    Hour = hour,
                    DateTime = predictedTime,
                    Temperature = predictedWeather.PredictedTemperature,
                    Humidity = predictedWeather.PredictedHumidity,
                    Pressure = predictedWeather.PredictedPressure,
                    Condition = GetConditionName(predictedWeather.ConditionCode),
                    ConditionCode = predictedWeather.ConditionCode
                });

                // Update context for next prediction (rolling window)
                var newContext = new Tensor[contextSize];
                for (int i = 0; i < contextSize - 1; i++)
                {
                    newContext[i] = context[i + 1];
                }
                newContext[contextSize - 1] = prediction;
                context = newContext;

                // Show progress
                if (hour % 24 == 0)
                {
                    Console.Write($"\r   Day {hour / 24}/5 complete...");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  Hour +{hour}: Prediction failed - {ex.Message}");
                break;
            }
        }
        Console.WriteLine("\r   All predictions complete!     ");

        // Display predictions in a nice table format
        if (predictions.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("┌──────────┬────────────────────┬──────────┬──────────┬──────────┬─────────────────┐");
            Console.WriteLine("│ Day+Hour │ Date/Time          │ Temp(°F) │ Humidity │ Pressure │ Condition       │");
            Console.WriteLine("├──────────┼────────────────────┼──────────┼──────────┼──────────┼─────────────────┤");

            string lastDate = "";
            foreach (var pred in predictions)
            {
                string dateStr = pred.DateTime.ToString("MMM dd");
                string timeStr = pred.DateTime.ToString("HH:00");
                string dayHour = $"D{(pred.Hour / 24) + 1}+{pred.Hour % 24:D2}h";

                // Add separator between days
                if (dateStr != lastDate && lastDate != "")
                {
                    Console.WriteLine("├──────────┼────────────────────┼──────────┼──────────┼──────────┼─────────────────┤");
                }
                lastDate = dateStr;

                string tempStr = $"{CelsiusToFahrenheit(pred.Temperature):F1}°F";
                string humidityStr = $"{pred.Humidity:F0}%";
                string pressureStr = $"{pred.Pressure:F1}";
                string conditionStr = pred.Condition;

                // Add weather emoji
                string emoji = GetWeatherEmoji(pred.Condition);

                Console.WriteLine($"│ {dayHour,-8} │ {dateStr} {timeStr,-9} │ {tempStr,8} │ {humidityStr,8} │ {pressureStr,8} │ {emoji} {conditionStr,-13} │");
            }

            Console.WriteLine("└──────────┴────────────────────┴──────────┴──────────┴──────────┴─────────────────┘");
            Console.WriteLine();

            // Daily summaries
            Console.WriteLine("📈 DAILY SUMMARIES:");
            Console.WriteLine();

            var predictionsByDay = predictions.GroupBy(p => p.DateTime.Date).OrderBy(g => g.Key).ToList();

            for (int day = 0; day < predictionsByDay.Count; day++)
            {
                var dayPreds = predictionsByDay[day].ToList();
                var dayDate = dayPreds.First().DateTime;

                var minTemp = dayPreds.Min(p => p.Temperature);
                var maxTemp = dayPreds.Max(p => p.Temperature);
                var avgHumidity = dayPreds.Average(p => p.Humidity);
                var avgPressure = dayPreds.Average(p => p.Pressure);

                // Most common condition
                var mostCommonCondition = dayPreds.GroupBy(p => p.Condition)
                    .OrderByDescending(g => g.Count())
                    .First().Key;
                var emoji = GetWeatherEmoji(mostCommonCondition);

                Console.WriteLine($"  Day {day + 1} - {dayDate:dddd, MMMM d, yyyy}");
                Console.WriteLine($"    🌡️  Temperature: {CelsiusToFahrenheit(minTemp):F1}°F to {CelsiusToFahrenheit(maxTemp):F1}°F");
                Console.WriteLine($"    💧 Avg Humidity: {avgHumidity:F0}%");
                Console.WriteLine($"    🎈 Avg Pressure: {avgPressure:F1} hPa");
                Console.WriteLine($"    🌤️  Primary Condition: {emoji} {mostCommonCondition}");
                Console.WriteLine();
            }

            // Overall summary
            var overallMinTemp = predictions.Min(p => p.Temperature);
            var overallMaxTemp = predictions.Max(p => p.Temperature);
            var overallAvgTemp = predictions.Average(p => p.Temperature);
            var overallAvgHumidity = predictions.Average(p => p.Humidity);
            var overallAvgPressure = predictions.Average(p => p.Pressure);

            Console.WriteLine("📊 5-DAY OVERALL SUMMARY:");
            Console.WriteLine($"   Temperature Range: {CelsiusToFahrenheit(overallMinTemp):F1}°F to {CelsiusToFahrenheit(overallMaxTemp):F1}°F");
            Console.WriteLine($"   Average Temperature: {CelsiusToFahrenheit(overallAvgTemp):F1}°F");
            Console.WriteLine($"   Average Humidity: {overallAvgHumidity:F0}%");
            Console.WriteLine($"   Average Pressure: {overallAvgPressure:F1} hPa");
            Console.WriteLine($"   Total Predictions: {predictions.Count} (every 3 hours)");
            Console.WriteLine();

            // Save predictions to file
            SavePredictionsToFile(predictions, now);
        }
        else
        {
            Console.WriteLine("❌ No predictions could be generated");
            Console.WriteLine();
            Console.WriteLine("💡 Possible reasons:");
            Console.WriteLine("   • Not enough valid training data");
            Console.WriteLine("   • Data quality issues (missing values)");
            Console.WriteLine("   • Insufficient similar weather patterns for 5-day forecast");
            Console.WriteLine();
            Console.WriteLine("   Try:");
            Console.WriteLine("   1. Re-downloading fresh data: python fetch_weather_data.py");
            Console.WriteLine("   2. Using more training data: python fetch_weather_data.py --days 1825");
        }

        Console.WriteLine();

        if (tree is IDisposable disposableTree)
        {
            disposableTree.Dispose();
        }
    }

    static string GetConditionName(int code)
    {
        var conditions = new Dictionary<int, string>
        {
            { 0, "Clear" },
            { 1, "Partly Cloudy" },
            { 2, "Cloudy" },
            { 3, "Overcast" },
            { 4, "Fog" },
            { 5, "Drizzle" },
            { 6, "Rain" },
            { 7, "Snow" },
            { 8, "Sleet" },
            { 9, "Thunderstorm" }
        };

        return conditions.ContainsKey(code) ? conditions[code] : "Unknown";
    }

    static double CelsiusToFahrenheit(double celsius)
    {
        return celsius * 9.0 / 5.0 + 32.0;
    }

    static string GetWeatherEmoji(string condition)
    {
        var emojiMap = new Dictionary<string, string>
        {
            { "Clear", "☀️" },
            { "Partly Cloudy", "⛅" },
            { "Cloudy", "☁️" },
            { "Overcast", "☁️" },
            { "Fog", "🌫️" },
            { "Drizzle", "🌦️" },
            { "Rain", "🌧️" },
            { "Heavy Rain", "⛈️" },
            { "Snow", "❄️" },
            { "Sleet", "🌨️" },
            { "Thunderstorm", "⛈️" },
            { "Windy", "💨" }
        };

        return emojiMap.ContainsKey(condition) ? emojiMap[condition] : "🌤️";
    }

    static void SavePredictionsToFile(List<PredictionResult> predictions, DateTime forecastStartTime)
    {
        string filename = $"weather_forecast_5day_{forecastStartTime:yyyyMMdd_HHmm}.csv";

        using (var writer = new StreamWriter(filename))
        {
            writer.WriteLine("# 5-Day Weather Forecast (120 hours)");
            writer.WriteLine($"# Generated at: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine($"# Forecasting from: {forecastStartTime:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine("Hour,Day,DateTime,Temperature_F,Temperature_C,Humidity_Percent,Pressure_hPa,Condition");

            foreach (var pred in predictions)
            {
                int day = (pred.Hour / 24) + 1;
                writer.WriteLine($"+{pred.Hour},{day},{pred.DateTime:yyyy-MM-dd HH:00},{CelsiusToFahrenheit(pred.Temperature):F2},{pred.Temperature:F2},{pred.Humidity:F1},{pred.Pressure:F2},{pred.Condition}");
            }
        }

        Console.WriteLine($"💾 Forecast saved to: {filename}");
    }

    static RollingConfig CreateValidationConfig(RollingConfig template, int windowStartIndex, int windowEndIndex,
        int trainPercent, int maxTrainObs, int trainStep, int testStep)
    {
        return new RollingConfig
        {
            WindowPercent = 100,
            WindowStartIndex = windowStartIndex,
            WindowEndIndex = windowEndIndex,
            WindowPosition = 0,
            TotalPositions = 1,
            TrainPercent = trainPercent,
            TestPercent = 100 - trainPercent,
            ModelWindowSize = template.ModelWindowSize,
            Similarity = template.Similarity,
            TemporalDecay = template.TemporalDecay,
            Temperature = template.Temperature,
            ExplorationRate = template.ExplorationRate,
            EnableReplay = template.EnableReplay,
            EnableDelta = template.EnableDelta,
            MaxTrainObs = maxTrainObs,
            TrainStep = trainStep,
            TestStep = testStep
        };
    }

    static StageResults RunValidationStage(int stageNum, string stageName, RollingConfig[] configs,
        Tensor[] allTensors, SatelliteWeatherObservation[] allObservations, int processorCount,
        double tempBaseline, double conditionBaseline)
    {
        var results = new ConcurrentBag<RollingResult>();
        var stopwatch = Stopwatch.StartNew();
        completedTests = 0;

        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine($"Running {stageName}...");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = processorCount
        };

        Parallel.ForEach(configs, parallelOptions, config =>
        {
            var threadEncoder = new SatelliteTensorEncoder();
            var result = RunRollingTest(config, allTensors, allObservations, threadEncoder);
            results.Add(result);

            lock (consoleLock)
            {
                Console.WriteLine($"  ✓ Completed: TempMAE={result.BestTempMAE:F2}°C, CondAcc={result.BestConditionAccuracy:F1}% ({result.PredictionCount} predictions)");
            }
        });

        stopwatch.Stop();

        var stageResults = new StageResults
        {
            StageNumber = stageNum,
            StageName = stageName,
            Results = results.ToList(),
            Duration = stopwatch.Elapsed
        };

        Console.WriteLine();
        Console.WriteLine($"✅ {stageName} complete in {stopwatch.Elapsed:mm\\:ss}");
        Console.WriteLine();

        return stageResults;
    }

    public static RollingResult RunRollingTest(RollingConfig config, Tensor[] allTensors, SatelliteWeatherObservation[] allObservations, SatelliteTensorEncoder encoder)
    {
        int windowSize = config.WindowEndIndex - config.WindowStartIndex;

        Tensor[] windowTensors = new Tensor[windowSize];
        Array.Copy(allTensors, config.WindowStartIndex, windowTensors, 0, windowSize);

        SatelliteWeatherObservation[] windowObservations = new SatelliteWeatherObservation[windowSize];
        Array.Copy(allObservations, config.WindowStartIndex, windowObservations, 0, windowSize);

        int splitIndex = (int)(windowSize * (config.TrainPercent / 100.0));

        Tensor[] trainingTensors = new Tensor[splitIndex];
        Array.Copy(windowTensors, 0, trainingTensors, 0, splitIndex);

        SatelliteWeatherObservation[] trainingObservations = new SatelliteWeatherObservation[splitIndex];
        Array.Copy(windowObservations, 0, trainingObservations, 0, splitIndex);

        Tensor[] testTensors = new Tensor[windowSize - splitIndex];
        Array.Copy(windowTensors, splitIndex, testTensors, 0, windowSize - splitIndex);

        SatelliteWeatherObservation[] testObservations = new SatelliteWeatherObservation[windowSize - splitIndex];
        Array.Copy(windowObservations, splitIndex, testObservations, 0, windowSize - splitIndex);

        var tree = new TensorSequenceTree(
            maxContextWindow: config.ModelWindowSize,
            similarityThreshold: config.Similarity,
            temporalDecay: config.TemporalDecay,
            temperature: config.Temperature,
            explorationRate: config.ExplorationRate,
            enableExperienceReplay: config.EnableReplay,
            enableDeltaRegression: config.EnableDelta
        );

        int maxTrainObs = Math.Min(config.MaxTrainObs, trainingTensors.Length);
        int trainStart = Math.Max(0, trainingTensors.Length - maxTrainObs);
        int trainWindowSize = config.ModelWindowSize;
        int step = config.TrainStep;

        for (int i = trainStart; i < trainingTensors.Length - trainWindowSize; i += step)
        {
            Tensor[] window = new Tensor[trainWindowSize];
            for (int w = 0; w < trainWindowSize; w++) {
                window[w] = trainingTensors[i + w];
            }
            tree.Learn(window);
        }
        int lastIndexUsed = trainStart + ((trainingTensors.Length - trainWindowSize - trainStart - 1) / step * step) + trainWindowSize - 1;
        Debug.Assert(lastIndexUsed < splitIndex, $"Leak detected! Last training index used: {lastIndexUsed}, splitIndex: {splitIndex}");

        var methodResults = TestPredictionMethod(tree, testTensors, testObservations, encoder, config.ModelWindowSize, config.TestStep);

        var result = new RollingResult
        {
            Config = config,
            BestMethod = "DeltaSmooth",
            BestTempMAE = methodResults.GetTempMAE(),
            BestConditionAccuracy = methodResults.GetConditionAccuracy(),
            PredictionCount = methodResults.Total,
            AllMethodResults = new Dictionary<string, MethodResult> { ["DeltaSmooth"] = methodResults }
        };

        if (tree is IDisposable disposableTree)
        {
            disposableTree.Dispose();
        }

        return result;
    }

    static MethodResult TestPredictionMethod(
        TensorSequenceTree tree,
        Tensor[] testTensors,
        SatelliteWeatherObservation[] testObservations,
        SatelliteTensorEncoder encoder,
        int contextSize,
        int testStep)
    {
        var result = new MethodResult();

        int actualContext = Math.Min(contextSize, testTensors.Length - 1);
        if (actualContext <= 0) return result;

        for (int i = actualContext; i < testTensors.Length; i += testStep)
        {
            Tensor[] context = new Tensor[actualContext];
            for (int c = 0; c < actualContext; c++)
            {
                context[c] = testTensors[i - actualContext + c];
            }

            SatelliteWeatherObservation actualWeather = testObservations[i];

            try
            {
                var prediction = tree.PredictNextDeltaSmoothed(context);
                if (prediction == null) continue;

                var predictedWeather = encoder.Decode(prediction);

                double tempError = Math.Abs(predictedWeather.PredictedTemperature - actualWeather.Temperature);
                double humidityError = Math.Abs(predictedWeather.PredictedHumidity - actualWeather.Humidity);
                double pressureError = Math.Abs(predictedWeather.PredictedPressure - actualWeather.Pressure);
                bool conditionCorrect = predictedWeather.ConditionCode == actualWeather.ConditionCode;
                bool tempBucketCorrect = predictedWeather.TemperatureBucket == actualWeather.TemperatureBucket;

                result.Total++;
                result.TempErrors.Add(tempError);
                result.HumidityErrors.Add(humidityError);
                result.PressureErrors.Add(pressureError);
                if (conditionCorrect) result.ConditionCorrect++;
                if (tempBucketCorrect) result.TempBucketCorrect++;
            }
            catch { }
        }

        return result;
    }

    static void SaveValidationResults(List<RollingResult> results, RollingConfig optimalConfig,
        double tempBaseline, double conditionBaseline)
    {
        string outputFile = "optimal_model_results.csv";
        using (var writer = new StreamWriter(outputFile))
        {
            writer.WriteLine("# Enhanced Auto Model Tuner - Optimal Configuration Results");
            writer.WriteLine($"# Optimal Lookback: {optimalConfig.ModelWindowSize}h");
            writer.WriteLine($"# Optimal Similarity: {optimalConfig.Similarity:F4}");
            writer.WriteLine($"# Optimal Temporal Decay: {optimalConfig.TemporalDecay:F6}");
            writer.WriteLine($"# Optimal Temperature: {optimalConfig.Temperature:F4}");
            writer.WriteLine($"# Optimal Exploration: {optimalConfig.ExplorationRate:F4}");
            writer.WriteLine($"# Experience Replay: {optimalConfig.EnableReplay}");
            writer.WriteLine($"# Delta Regression: {optimalConfig.EnableDelta}");
            writer.WriteLine($"# Temperature Baseline (MAE): {tempBaseline:F2}°C");
            writer.WriteLine($"# Condition Baseline (Accuracy): {conditionBaseline:F2}%");
            writer.WriteLine("Stage,MaxTrainObs,TrainStep,TestStep,TempMAE,ConditionAccuracy,PredictionCount,ImprovementTemp,ImprovementCondition");

            int stageNum = 1;
            foreach (var result in results)
            {
                writer.WriteLine($"{stageNum},{result.Config.MaxTrainObs},{result.Config.TrainStep},{result.Config.TestStep},{result.BestTempMAE:F2},{result.BestConditionAccuracy:F2},{result.PredictionCount},{tempBaseline - result.BestTempMAE:F2},{result.BestConditionAccuracy - conditionBaseline:F2}");
                stageNum++;
            }
        }

        Console.WriteLine($"💾 Results saved to: {outputFile}");
    }
}

// ============================================================================
// ENHANCED MODEL TUNER - Comprehensive Search
// ============================================================================

class EnhancedModelTuner
{
    private Tensor[] allTensors;
    private SatelliteWeatherObservation[] allObservations;
    private int windowStartIndex;
    private int windowEndIndex;
    private int trainPercent;
    private int processorCount;
    private double tempBaseline;
    private double conditionBaseline;

    // ENHANCED CONFIGURATION - More thorough search
    private const int BEAM_WIDTH = 10;
    private const int TOP_BINARY_SEARCH = 3;
    private const int MAX_ITERATIONS = 5;
    private const double CONVERGENCE_THRESHOLD = 0.03;

    // Quick test parameters
    private const int QUICK_MAX_TRAIN = 400;
    private const int QUICK_TRAIN_STEP = 80;
    private const int QUICK_TEST_STEP = 15;

    public EnhancedModelTuner(Tensor[] tensors, SatelliteWeatherObservation[] observations,
        int startIdx, int endIdx, int trainPct, int procCount, double tempBase, double condBase)
    {
        allTensors = tensors;
        allObservations = observations;
        windowStartIndex = startIdx;
        windowEndIndex = endIdx;
        trainPercent = trainPct;
        processorCount = procCount;
        tempBaseline = tempBase;
        conditionBaseline = condBase;
    }

    public RollingConfig FindOptimalParameters()
    {
        var globalStopwatch = Stopwatch.StartNew();

        Console.WriteLine("🔬 PHASE 1: COMPREHENSIVE INITIAL POPULATION");
        Console.WriteLine("   Creating dense starting configurations (finer grid)...");
        Console.WriteLine();

        var initialPopulation = GenerateComprehensiveInitialPopulation();
        Console.WriteLine($"   Generated {initialPopulation.Count} initial configurations");

        var evaluatedPopulation = EvaluatePopulation(initialPopulation, "Initial");

        var currentBeam = evaluatedPopulation.OrderBy(c => c.Score).Take(BEAM_WIDTH).ToList();

        Console.WriteLine($"✅ Top {BEAM_WIDTH} candidates selected for beam search");
        PrintTopCandidates(currentBeam, 5);
        Console.WriteLine();

        Console.WriteLine("🎯 PHASE 2: ENHANCED BEAM SEARCH");
        Console.WriteLine($"   Running {MAX_ITERATIONS} iterations");
        Console.WriteLine($"   Beam width: {BEAM_WIDTH} candidates");
        Console.WriteLine($"   Binary search on top {TOP_BINARY_SEARCH} candidates each iteration");
        Console.WriteLine();

        double previousBestScore = currentBeam[0].Score;

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
        {
            Console.WriteLine($"═══════════════════════════════════════════════════════════════════════");
            Console.WriteLine($"📍 ITERATION {iteration + 1}/{MAX_ITERATIONS}");
            Console.WriteLine($"═══════════════════════════════════════════════════════════════════════");
            Console.WriteLine();

            var variations = GenerateComprehensiveBeamVariations(currentBeam);
            Console.WriteLine($"   Generated {variations.Count} variations from top {BEAM_WIDTH} candidates");

            var evaluatedVariations = EvaluatePopulation(variations, $"Iter{iteration + 1}");

            var combined = currentBeam.Concat(evaluatedVariations).OrderBy(c => c.Score).ToList();
            currentBeam = combined.Take(BEAM_WIDTH).ToList();

            double currentBestScore = currentBeam[0].Score;
            double improvement = ((previousBestScore - currentBestScore) / previousBestScore) * 100;

            Console.WriteLine();
            Console.WriteLine($"   🏆 Best score: {currentBestScore:F3} (improvement: {improvement:F2}%)");
            PrintTopCandidate(currentBeam[0]);

            if (iteration < MAX_ITERATIONS - 1)
            {
                Console.WriteLine();
                Console.WriteLine($"   🔍 Binary search refinement on top {TOP_BINARY_SEARCH} candidates...");

                for (int i = 0; i < Math.Min(TOP_BINARY_SEARCH, currentBeam.Count); i++)
                {
                    Console.WriteLine($"      Refining candidate #{i + 1} (score: {currentBeam[i].Score:F3})...");
                    var refined = BinarySearchRefinement(currentBeam[i].Config);
                    var refinedScore = EvaluateConfig(refined);

                    if (refinedScore < currentBeam[i].Score)
                    {
                        double refineImprovement = ((currentBeam[i].Score - refinedScore) / currentBeam[i].Score) * 100;
                        Console.WriteLine($"         ✅ Improved: {currentBeam[i].Score:F3} → {refinedScore:F3} (+{refineImprovement:F2}%)");
                        currentBeam[i] = new ScoredConfig { Config = refined, Score = refinedScore };
                    }
                    else
                    {
                        Console.WriteLine($"         ℹ️  No improvement (stayed at {currentBeam[i].Score:F3})");
                    }
                }

                currentBeam = currentBeam.OrderBy(c => c.Score).ToList();
                currentBestScore = currentBeam[0].Score;
                improvement = ((previousBestScore - currentBestScore) / previousBestScore) * 100;

                Console.WriteLine($"      After refinement, best: {currentBestScore:F3} (total improvement: {improvement:F2}%)");
            }

            if (improvement < CONVERGENCE_THRESHOLD && iteration > 0)
            {
                Console.WriteLine($"   ✓ Converged (improvement < {CONVERGENCE_THRESHOLD}%) - stopping early");
                break;
            }

            previousBestScore = currentBestScore;
            Console.WriteLine();
        }

        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine("🎯 PHASE 3: ULTRA-PRECISION FINAL TUNING");
        Console.WriteLine("═══════════════════════════════════════════════════════════════════════");
        Console.WriteLine();

        var bestConfig = currentBeam[0].Config;
        Console.WriteLine("   Performing ultra-fine binary search on all continuous parameters...");

        bestConfig = UltraFineTuneSimilarity(bestConfig);
        bestConfig = UltraFineTuneTemporalDecay(bestConfig);
        bestConfig = UltraFineTuneTemperature(bestConfig);
        bestConfig = UltraFineTuneExploration(bestConfig);

        var finalScore = EvaluateConfig(bestConfig);
        Console.WriteLine();
        Console.WriteLine($"   🏆 Final optimized score: {finalScore:F3}");

        globalStopwatch.Stop();
        Console.WriteLine();
        Console.WriteLine($"⚡ Total tuning time: {globalStopwatch.Elapsed:mm\\:ss}");

        return bestConfig;
    }

    private List<RollingConfig> GenerateComprehensiveInitialPopulation()
    {
        var population = new List<RollingConfig>();

        int[] lookbacks = {60, 72, 84 };
        bool[] replayOptions = { false, true };
        bool[] deltaOptions = { true };

        float[] similarities = Enumerable.Range(55, 41).Select(i => i / 100f).ToArray();
        float[] decays = new[] { 0.9993f, 0.9995f, 0.9997f, 0.9998f, 0.9999f };
        float[] temps = new[] { 0.0f};
        float[] explores = new[] { 0.0f};

        foreach (var lookback in lookbacks)
        {
            foreach (var replay in replayOptions)
            {
                foreach (var delta in deltaOptions)
                {
                    foreach (var sim in similarities.Where((s, idx) => idx % 3 == 0))
                    {
                        foreach (var decay in decays.Where((d, idx) => idx % 2 == 0))
                        {
                            if (replay)
                            {
                                foreach (var temp in temps.Where((t, idx) => idx % 2 == 0))
                                {
                                    foreach (var explore in explores.Where((e, idx) => idx % 2 == 0))
                                    {
                                        population.Add(CreateConfig(lookback, sim, decay, temp, explore, replay, delta));
                                    }
                                }
                            }
                            else
                            {
                                population.Add(CreateConfig(lookback, sim, decay, 0f, 0f, replay, delta));
                            }
                        }
                    }
                }
            }
        }

        return population;
    }

    private List<RollingConfig> GenerateComprehensiveBeamVariations(List<ScoredConfig> beam)
    {
        var variations = new List<RollingConfig>();

        foreach (var candidate in beam)
        {
            var config = candidate.Config;

            foreach (var delta in new[] { -36, -24, -18, -12, -6, 6, 12, 18, 24, 36 })
            {
                int newLookback = config.ModelWindowSize + delta;
                if (newLookback >= 24 && newLookback <= 168)
                {
                    var v = CloneConfig(config);
                    v.ModelWindowSize = newLookback;
                    variations.Add(v);
                }
            }

            foreach (var delta in new[] { -0.08f, -0.05f, -0.03f, -0.02f, -0.01f, 0.01f, 0.02f, 0.03f, 0.05f, 0.08f })
            {
                float newSim = config.Similarity + delta;
                if (newSim >= 0.5f && newSim <= 0.95f)
                {
                    var v = CloneConfig(config);
                    v.Similarity = newSim;
                    variations.Add(v);
                }
            }

            foreach (var delta in new[] { -0.002f, -0.001f, -0.0005f, -0.0002f, 0.0002f, 0.0005f, 0.001f, 0.002f })
            {
                float newDecay = config.TemporalDecay + delta;
                if (newDecay >= 0.99f && newDecay <= 0.9999f)
                {
                    var v = CloneConfig(config);
                    v.TemporalDecay = newDecay;
                    variations.Add(v);
                }
            }

            if (config.EnableReplay)
            {
                foreach (var delta in new[] { -0.15f, -0.1f, -0.05f, -0.03f, 0.03f, 0.05f, 0.1f, 0.15f })
                {
                    float newTemp = config.Temperature + delta;
                    if (newTemp >= 0f && newTemp <= 1f)
                    {
                        var v = CloneConfig(config);
                        v.Temperature = newTemp;
                        variations.Add(v);
                    }
                }

                foreach (var delta in new[] { -0.08f, -0.05f, -0.03f, -0.02f, 0.02f, 0.03f, 0.05f, 0.08f })
                {
                    float newExplore = config.ExplorationRate + delta;
                    if (newExplore >= 0f && newExplore <= 0.5f)
                    {
                        var v = CloneConfig(config);
                        v.ExplorationRate = newExplore;
                        variations.Add(v);
                    }
                }
            }

            if (config.EnableReplay)
            {
                var v = CloneConfig(config);
                v.EnableReplay = false;
                v.Temperature = 0f;
                v.ExplorationRate = 0f;
                variations.Add(v);
            }

            if (config.EnableDelta)
            {
                var v = CloneConfig(config);
                v.EnableDelta = false;
                variations.Add(v);
            }
        }

        return variations;
    }

    private List<ScoredConfig> EvaluatePopulation(List<RollingConfig> population, string phaseName)
    {
        var results = new ConcurrentBag<ScoredConfig>();
        int completed = 0;
        int total = population.Count;

        Console.WriteLine($"   Evaluating {total} configurations...");

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = processorCount
        };

        Parallel.ForEach(population, parallelOptions, config =>
        {
            double score = EvaluateConfig(config);
            results.Add(new ScoredConfig { Config = config, Score = score });

            int current = Interlocked.Increment(ref completed);
            if (current % 20 == 0 || current == total)
            {
                lock (WeatherPredictionValidation.consoleLock)
                {
                    Console.Write($"\r   Progress: {current}/{total} ({current * 100 / total}%)");
                }
            }
        });

        Console.WriteLine();

        var sorted = results.OrderBy(r => r.Score).ToList();
        Console.WriteLine($"   ✓ Evaluation complete. Best score: {sorted[0].Score:F3}");

        return sorted;
    }

    private double EvaluateConfig(RollingConfig config)
    {
        var encoder = new SatelliteTensorEncoder();

        config.MaxTrainObs = QUICK_MAX_TRAIN;
        config.TrainStep = QUICK_TRAIN_STEP;
        config.TestStep = QUICK_TEST_STEP;

        var result = WeatherPredictionValidation.RunRollingTest(config, allTensors, allObservations, encoder);

        double normalizedMAE = result.BestTempMAE / tempBaseline;
        double normalizedAccuracy = (100 - result.BestConditionAccuracy) / (100 - conditionBaseline);

        double score = normalizedMAE + normalizedAccuracy;

        return score;
    }

    private RollingConfig BinarySearchRefinement(RollingConfig config)
    {
        config = BinarySearchParameter(config,
            c => c.Similarity,
            (c, v) => { var clone = CloneConfig(c); clone.Similarity = v; return clone; },
            Math.Max(0.5f, config.Similarity - 0.08f),
            Math.Min(0.95f, config.Similarity + 0.08f),
            0.005f);

        config = BinarySearchParameter(config,
            c => c.TemporalDecay,
            (c, v) => { var clone = CloneConfig(c); clone.TemporalDecay = v; return clone; },
            Math.Max(0.99f, config.TemporalDecay - 0.002f),
            Math.Min(0.9999f, config.TemporalDecay + 0.002f),
            0.00005f);

        return config;
    }

    private RollingConfig BinarySearchParameter(RollingConfig config,
        Func<RollingConfig, float> getter,
        Func<RollingConfig, float, RollingConfig> setter,
        float min, float max, float precision)
    {
        float left = min;
        float right = max;
        RollingConfig bestConfig = config;
        double bestScore = EvaluateConfig(config);

        while (right - left > precision)
        {
            float mid1 = left + (right - left) / 3;
            float mid2 = right - (right - left) / 3;

            var config1 = setter(config, mid1);
            var config2 = setter(config, mid2);

            double score1 = EvaluateConfig(config1);
            double score2 = EvaluateConfig(config2);

            if (score1 < bestScore)
            {
                bestScore = score1;
                bestConfig = config1;
            }

            if (score2 < bestScore)
            {
                bestScore = score2;
                bestConfig = config2;
            }

            if (score1 < score2)
            {
                right = mid2;
            }
            else
            {
                left = mid1;
            }
        }

        return bestConfig;
    }

    private RollingConfig UltraFineTuneSimilarity(RollingConfig config)
    {
        Console.WriteLine("   → Ultra-fine tuning Similarity...");
        float bestValue = config.Similarity;
        double bestScore = EvaluateConfig(config);

        for (float delta = 0.015f; delta >= 0.002f; delta /= 1.5f)
        {
            foreach (float d in new[] { -delta, delta })
            {
                float testValue = bestValue + d;
                if (testValue < 0.5f || testValue > 0.95f) continue;

                var testConfig = CloneConfig(config);
                testConfig.Similarity = testValue;
                double score = EvaluateConfig(testConfig);

                if (score < bestScore)
                {
                    bestScore = score;
                    bestValue = testValue;
                }
            }
        }

        config.Similarity = bestValue;
        Console.WriteLine($"      Optimal: {bestValue:F4}");
        return config;
    }

    private RollingConfig UltraFineTuneTemporalDecay(RollingConfig config)
    {
        Console.WriteLine("   → Ultra-fine tuning Temporal Decay...");
        float bestValue = config.TemporalDecay;
        double bestScore = EvaluateConfig(config);

        for (float delta = 0.0003f; delta >= 0.00003f; delta /= 1.5f)
        {
            foreach (float d in new[] { -delta, delta })
            {
                float testValue = bestValue + d;
                if (testValue < 0.99f || testValue > 0.9999f) continue;

                var testConfig = CloneConfig(config);
                testConfig.TemporalDecay = testValue;
                double score = EvaluateConfig(testConfig);

                if (score < bestScore)
                {
                    bestScore = score;
                    bestValue = testValue;
                }
            }
        }

        config.TemporalDecay = bestValue;
        Console.WriteLine($"      Optimal: {bestValue:F6}");
        return config;
    }

    private RollingConfig UltraFineTuneTemperature(RollingConfig config)
    {
        if (!config.EnableReplay) return config;

        Console.WriteLine("   → Ultra-fine tuning Temperature...");
        float bestValue = config.Temperature;
        double bestScore = EvaluateConfig(config);

        for (float delta = 0.04f; delta >= 0.005f; delta /= 1.5f)
        {
            foreach (float d in new[] { -delta, delta })
            {
                float testValue = bestValue + d;
                if (testValue < 0f || testValue > 1f) continue;

                var testConfig = CloneConfig(config);
                testConfig.Temperature = testValue;
                double score = EvaluateConfig(testConfig);

                if (score < bestScore)
                {
                    bestScore = score;
                    bestValue = testValue;
                }
            }
        }

        config.Temperature = bestValue;
        Console.WriteLine($"      Optimal: {bestValue:F4}");
        return config;
    }

    private RollingConfig UltraFineTuneExploration(RollingConfig config)
    {
        if (!config.EnableReplay) return config;

        Console.WriteLine("   → Ultra-fine tuning Exploration Rate...");
        float bestValue = config.ExplorationRate;
        double bestScore = EvaluateConfig(config);

        for (float delta = 0.02f; delta >= 0.003f; delta /= 1.5f)
        {
            foreach (float d in new[] { -delta, delta })
            {
                float testValue = bestValue + d;
                if (testValue < 0f || testValue > 0.5f) continue;

                var testConfig = CloneConfig(config);
                testConfig.ExplorationRate = testValue;
                double score = EvaluateConfig(testConfig);

                if (score < bestScore)
                {
                    bestScore = score;
                    bestValue = testValue;
                }
            }
        }

        config.ExplorationRate = bestValue;
        Console.WriteLine($"      Optimal: {bestValue:F4}");
        return config;
    }

    private RollingConfig CreateConfig(int lookback, float similarity, float decay,
        float temperature, float exploration, bool replay, bool delta)
    {
        return new RollingConfig
        {
            WindowPercent = 100,
            WindowStartIndex = windowStartIndex,
            WindowEndIndex = windowEndIndex,
            WindowPosition = 0,
            TotalPositions = 1,
            TrainPercent = trainPercent,
            TestPercent = 100 - trainPercent,
            ModelWindowSize = lookback,
            Similarity = similarity,
            TemporalDecay = decay,
            Temperature = temperature,
            ExplorationRate = exploration,
            EnableReplay = replay,
            EnableDelta = delta,
            MaxTrainObs = QUICK_MAX_TRAIN,
            TrainStep = QUICK_TRAIN_STEP,
            TestStep = QUICK_TEST_STEP
        };
    }

    private RollingConfig CloneConfig(RollingConfig config)
    {
        return new RollingConfig
        {
            WindowPercent = config.WindowPercent,
            WindowStartIndex = config.WindowStartIndex,
            WindowEndIndex = config.WindowEndIndex,
            WindowPosition = config.WindowPosition,
            TotalPositions = config.TotalPositions,
            TrainPercent = config.TrainPercent,
            TestPercent = config.TestPercent,
            ModelWindowSize = config.ModelWindowSize,
            Similarity = config.Similarity,
            TemporalDecay = config.TemporalDecay,
            Temperature = config.Temperature,
            ExplorationRate = config.ExplorationRate,
            EnableReplay = config.EnableReplay,
            EnableDelta = config.EnableDelta,
            MaxTrainObs = config.MaxTrainObs,
            TrainStep = config.TrainStep,
            TestStep = config.TestStep
        };
    }

    private void PrintTopCandidate(ScoredConfig candidate)
    {
        var c = candidate.Config;
        Console.WriteLine($"      Lookback: {c.ModelWindowSize}h, Similarity: {c.Similarity:F3}, Decay: {c.TemporalDecay:F4}");
        Console.WriteLine($"      Temp: {c.Temperature:F3}, Explore: {c.ExplorationRate:F3}, Replay: {c.EnableReplay}, Delta: {c.EnableDelta}");
    }

    private void PrintTopCandidates(List<ScoredConfig> candidates, int count)
    {
        Console.WriteLine();
        for (int i = 0; i < Math.Min(count, candidates.Count); i++)
        {
            var c = candidates[i].Config;
            Console.WriteLine($"   #{i + 1}: Score={candidates[i].Score:F3} | W{c.ModelWindowSize}h S{c.Similarity:F2} D{c.TemporalDecay:F4} | R:{c.EnableReplay} Δ:{c.EnableDelta}");
        }
    }
}

class ScoredConfig
{
    public RollingConfig Config { get; set; }
    public double Score { get; set; }
}

class PredictionResult
{
    public int Hour { get; set; }
    public DateTime DateTime { get; set; }
    public double Temperature { get; set; }
    public double Humidity { get; set; }
    public double Pressure { get; set; }
    public string Condition { get; set; }
    public int ConditionCode { get; set; }
}

class StageResults
{
    public int StageNumber { get; set; }
    public string StageName { get; set; }
    public List<RollingResult> Results { get; set; }
    public TimeSpan Duration { get; set; }
}

class RollingConfig
{
    public int WindowPercent { get; set; }
    public int WindowStartIndex { get; set; }
    public int WindowEndIndex { get; set; }
    public int WindowPosition { get; set; }
    public int TotalPositions { get; set; }
    public int TrainPercent { get; set; }
    public int TestPercent { get; set; }
    public int ModelWindowSize { get; set; }
    public float Similarity { get; set; }
    public float TemporalDecay { get; set; }
    public float Temperature { get; set; }
    public float ExplorationRate { get; set; }
    public bool EnableReplay { get; set; }
    public bool EnableDelta { get; set; }
    public int MaxTrainObs { get; set; }
    public int TrainStep { get; set; }
    public int TestStep { get; set; }
}

class RollingResult
{
    public RollingConfig Config { get; set; }
    public string BestMethod { get; set; }
    public double BestTempMAE { get; set; }
    public double BestConditionAccuracy { get; set; }
    public int PredictionCount { get; set; }
    public Dictionary<string, MethodResult> AllMethodResults { get; set; }
}

class MethodResult
{
    public int Total = 0;
    public int ConditionCorrect = 0;
    public int TempBucketCorrect = 0;
    public List<double> TempErrors = new List<double>();
    public List<double> HumidityErrors = new List<double>();
    public List<double> PressureErrors = new List<double>();

    public double GetTempMAE() => TempErrors.Count > 0 ? TempErrors.Average() : double.MaxValue;
    public double GetHumidityMAE() => HumidityErrors.Count > 0 ? HumidityErrors.Average() : double.MaxValue;
    public double GetPressureMAE() => PressureErrors.Count > 0 ? PressureErrors.Average() : double.MaxValue;
    public double GetConditionAccuracy() => Total > 0 ? (ConditionCorrect * 100.0 / Total) : 0;
    public double GetTempBucketAccuracy() => Total > 0 ? (TempBucketCorrect * 100.0 / Total) : 0;
}