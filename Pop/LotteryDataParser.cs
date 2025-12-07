using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using repliKate;

// ============================================================================
// LOTTERY METADATA ANALYSIS - Extract temporal patterns from draw history
// ============================================================================
// CSV Format: 2024,"Tuesday - 11:59pmDecember 31, 2024",225,13
// Fields: Year, DayTime String, Draw Number, Lottery Number
// ============================================================================

public class LotteryDraw
{
    public int Year { get; set; }
    public string DateTimeString { get; set; }
    public int DrawNumber { get; set; }
    public int LotteryNumber { get; set; }

    // Extracted features
    public DayOfWeek DayOfWeek { get; set; }
    public int Month { get; set; }
    public int Day { get; set; }
    public int Hour { get; set; }
    public bool IsWeekend { get; set; }
    public bool IsEvening { get; set; }  // After 6pm
    public bool IsLateNight { get; set; } // After 10pm

    // Historical features (calculated later)
    public int DaysSinceLastAppearance { get; set; }
    public double FrequencyInLast50 { get; set; }
    public int ConsecutiveDrawsWithoutAppearing { get; set; }

    public override string ToString()
    {
        return $"Draw {DrawNumber}: Number {LotteryNumber} on {DayOfWeek} {Month}/{Day} at {Hour:D2}:00";
    }
}

public class LotteryDataParser
{
    public static List<LotteryDraw> ParseCSV(string filePath)
    {
        var draws = new List<LotteryDraw>();

        Console.WriteLine("📖 Parsing lottery data...");

        var lines = File.ReadAllLines(filePath);
        int parsed = 0, failed = 0;

        foreach (var line in lines)
        {
            try
            {
                var draw = ParseLine(line);
                if (draw != null)
                {
                    draws.Add(draw);
                    parsed++;
                }
            }
            catch (Exception ex)
            {
                failed++;
                if (failed <= 5) // Show first 5 errors
                {
                    Console.WriteLine($"⚠️  Parse error: {ex.Message} | Line: {line.Substring(0, Math.Min(50, line.Length))}...");
                }
            }
        }

        Console.WriteLine($"✅ Parsed {parsed} draws, {failed} errors");
        Console.WriteLine();

        return draws;
    }

    private static LotteryDraw ParseLine(string line)
    {
        // Format: 2024,"Tuesday - 11:59pmDecember 31, 2024",225,13

        // Split by comma, but respect quotes
        var parts = SplitCSV(line);

        if (parts.Length < 4)
        {
            throw new Exception($"Expected 4 fields, got {parts.Length}");
        }

        var draw = new LotteryDraw();

        // Parse year
        draw.Year = int.Parse(parts[0].Trim());

        // Parse datetime string
        draw.DateTimeString = parts[1].Trim().Trim('"');
        ExtractDateTimeFeatures(draw);

        // Parse draw number
        draw.DrawNumber = int.Parse(parts[2].Trim());

        // Parse lottery number
        draw.LotteryNumber = int.Parse(parts[3].Trim());

        return draw;
    }

    private static void ExtractDateTimeFeatures(LotteryDraw draw)
    {
        // Example: "Tuesday - 11:59pmDecember 31, 2024"
        var dateTimeStr = draw.DateTimeString;

        // Extract day of week
        var dayMatch = Regex.Match(dateTimeStr, @"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)");
        if (dayMatch.Success)
        {
            draw.DayOfWeek = Enum.Parse<DayOfWeek>(dayMatch.Value);
            draw.IsWeekend = (draw.DayOfWeek == DayOfWeek.Saturday || draw.DayOfWeek == DayOfWeek.Sunday);
        }

        // Extract time (e.g., "11:59pm")
        var timeMatch = Regex.Match(dateTimeStr, @"(\d{1,2}):(\d{2})(am|pm)", RegexOptions.IgnoreCase);
        if (timeMatch.Success)
        {
            int hour = int.Parse(timeMatch.Groups[1].Value);
            string ampm = timeMatch.Groups[3].Value.ToLower();

            if (ampm == "pm" && hour != 12)
                hour += 12;
            else if (ampm == "am" && hour == 12)
                hour = 0;

            draw.Hour = hour;
            draw.IsEvening = hour >= 18;  // 6pm or later
            draw.IsLateNight = hour >= 22; // 10pm or later
        }

        // Extract month and day (e.g., "December 31, 2024")
        var dateMatch = Regex.Match(dateTimeStr, @"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})");
        if (dateMatch.Success)
        {
            string monthName = dateMatch.Groups[1].Value;
            draw.Month = DateTime.ParseExact(monthName, "MMMM", CultureInfo.InvariantCulture).Month;
            draw.Day = int.Parse(dateMatch.Groups[2].Value);
        }
    }

    private static string[] SplitCSV(string line)
    {
        var result = new List<string>();
        var current = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];

            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                result.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }

        result.Add(current.ToString());
        return result.ToArray();
    }

    public static void CalculateHistoricalFeatures(List<LotteryDraw> draws)
    {
        Console.WriteLine("🔍 Calculating historical features...");

        // Track last appearance of each number
        var lastAppearance = new Dictionary<int, int>();

        // Calculate for each draw
        for (int i = 0; i < draws.Count; i++)
        {
            var draw = draws[i];
            int number = draw.LotteryNumber;

            // Days since last appearance
            if (lastAppearance.ContainsKey(number))
            {
                draw.DaysSinceLastAppearance = i - lastAppearance[number];
                draw.ConsecutiveDrawsWithoutAppearing = i - lastAppearance[number];
            }
            else
            {
                draw.DaysSinceLastAppearance = i; // First appearance
                draw.ConsecutiveDrawsWithoutAppearing = i;
            }

            // Frequency in last 50 draws
            int startIdx = Math.Max(0, i - 50);
            int count = 0;
            for (int j = startIdx; j < i; j++)
            {
                if (draws[j].LotteryNumber == number)
                    count++;
            }
            draw.FrequencyInLast50 = count / (double)Math.Min(50, i);

            // Update last appearance
            lastAppearance[number] = i;
        }

        Console.WriteLine("✅ Historical features calculated");
        Console.WriteLine();
    }
}

public class EnhancedTensorEncoder
{
    private int maxNumber;

    // Tensor layout:
    // [0-14]     : Number one-hot (15 numbers)
    // [15-21]    : Day of week one-hot (7 days)
    // [22-33]    : Month one-hot (12 months)
    // [34-57]    : Hour one-hot (24 hours)
    // [58]       : IsWeekend (0 or 1)
    // [59]       : IsEvening (0 or 1)
    // [60]       : IsLateNight (0 or 1)
    // [61]       : DaysSinceLastAppearance (normalized 0-1)
    // [62]       : FrequencyInLast50 (0-1)
    // [63]       : ConsecutiveDrawsWithoutAppearing (normalized 0-1)
    // Total: 64 dimensions

    public const int TENSOR_SIZE = 64;

    public EnhancedTensorEncoder(int maxNumber)
    {
        this.maxNumber = maxNumber;
    }

    public Tensor Encode(LotteryDraw draw)
    {
        float[] data = new float[TENSOR_SIZE];

        // Number one-hot encoding
        if (draw.LotteryNumber >= 1 && draw.LotteryNumber <= maxNumber)
        {
            data[draw.LotteryNumber - 1] = 1.0f;
        }

        // Day of week one-hot (15-21)
        data[15 + (int)draw.DayOfWeek] = 1.0f;

        // Month one-hot (22-33)
        if (draw.Month >= 1 && draw.Month <= 12)
        {
            data[22 + draw.Month - 1] = 1.0f;
        }

        // Hour one-hot (34-57)
        if (draw.Hour >= 0 && draw.Hour < 24)
        {
            data[34 + draw.Hour] = 1.0f;
        }

        // Boolean features (58-60)
        data[58] = draw.IsWeekend ? 1.0f : 0.0f;
        data[59] = draw.IsEvening ? 1.0f : 0.0f;
        data[60] = draw.IsLateNight ? 1.0f : 0.0f;

        // Historical features (61-63) - normalized
        data[61] = Math.Min(1.0f, draw.DaysSinceLastAppearance / 100.0f);
        data[62] = (float)draw.FrequencyInLast50;
        data[63] = Math.Min(1.0f, draw.ConsecutiveDrawsWithoutAppearing / 50.0f);

        return new Tensor(data);
    }

    public int Decode(Tensor tensor)
    {
        // Extract number from tensor (first 15 dimensions)
        int maxIndex = 0;
        float maxValue = tensor.Data[0];

        for (int i = 1; i < maxNumber; i++)
        {
            if (tensor.Data[i] > maxValue)
            {
                maxValue = tensor.Data[i];
                maxIndex = i;
            }
        }

        return maxIndex + 1;
    }

    public string DescribeFeatures(Tensor tensor)
    {
        var sb = new StringBuilder();

        // Number
        int number = Decode(tensor);
        sb.Append($"Number: {number}");

        // Day of week
        for (int i = 0; i < 7; i++)
        {
            if (tensor.Data[15 + i] > 0.5f)
            {
                sb.Append($", Day: {(DayOfWeek)i}");
                break;
            }
        }

        // Month
        for (int i = 0; i < 12; i++)
        {
            if (tensor.Data[22 + i] > 0.5f)
            {
                sb.Append($", Month: {i + 1}");
                break;
            }
        }

        // Hour
        for (int i = 0; i < 24; i++)
        {
            if (tensor.Data[34 + i] > 0.5f)
            {
                sb.Append($", Hour: {i}");
                break;
            }
        }

        // Flags
        if (tensor.Data[58] > 0.5f) sb.Append(", Weekend");
        if (tensor.Data[59] > 0.5f) sb.Append(", Evening");
        if (tensor.Data[60] > 0.5f) sb.Append(", LateNight");

        return sb.ToString();
    }
}

public class LotteryStatistics
{
    public static void PrintDatasetStatistics(List<LotteryDraw> draws)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                        DATASET STATISTICS                              ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        var numbers = draws.Select(d => d.LotteryNumber).ToList();
        var uniqueNumbers = numbers.Distinct().OrderBy(n => n).ToList();

        Console.WriteLine($"📊 Total Draws: {draws.Count}");
        Console.WriteLine($"🎲 Unique Numbers: {uniqueNumbers.Count} (Range: {uniqueNumbers.Min()}-{uniqueNumbers.Max()})");
        Console.WriteLine();

        // Frequency distribution
        Console.WriteLine("🔢 NUMBER FREQUENCY:");
        var frequency = numbers.GroupBy(n => n)
            .OrderByDescending(g => g.Count())
            .ToList();

        int topN = Math.Min(5, frequency.Count);
        Console.WriteLine($"  Top {topN} Most Frequent:");
        for (int i = 0; i < topN; i++)
        {
            var f = frequency[i];
            double percent = (f.Count() * 100.0) / draws.Count;
            string bar = new string('█', (int)(percent * 2));
            Console.WriteLine($"    #{f.Key}: {f.Count(),4} times ({percent,5:F2}%) {bar}");
        }
        Console.WriteLine();

        // Day of week distribution
        Console.WriteLine("📅 DAY OF WEEK DISTRIBUTION:");
        var dayFreq = draws.GroupBy(d => d.DayOfWeek)
            .OrderByDescending(g => g.Count())
            .ToList();

        foreach (var day in dayFreq)
        {
            double percent = (day.Count() * 100.0) / draws.Count;
            string bar = new string('█', (int)(percent / 2));
            Console.WriteLine($"  {day.Key,-10}: {day.Count(),4} draws ({percent,5:F2}%) {bar}");
        }
        Console.WriteLine();

        // Time of day distribution
        Console.WriteLine("🕐 TIME OF DAY:");
        int morningCount = draws.Count(d => d.Hour >= 6 && d.Hour < 12);
        int afternoonCount = draws.Count(d => d.Hour >= 12 && d.Hour < 18);
        int eveningCount = draws.Count(d => d.Hour >= 18 && d.Hour < 22);
        int lateNightCount = draws.Count(d => d.Hour >= 22 || d.Hour < 6);

        Console.WriteLine($"  Morning (6am-12pm):    {morningCount,4} ({morningCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine($"  Afternoon (12pm-6pm):  {afternoonCount,4} ({afternoonCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine($"  Evening (6pm-10pm):    {eveningCount,4} ({eveningCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine($"  Late Night (10pm-6am): {lateNightCount,4} ({lateNightCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine();

        // Weekend vs Weekday
        int weekendCount = draws.Count(d => d.IsWeekend);
        int weekdayCount = draws.Count - weekendCount;
        Console.WriteLine("📆 WEEKEND vs WEEKDAY:");
        Console.WriteLine($"  Weekday: {weekdayCount,4} ({weekdayCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine($"  Weekend: {weekendCount,4} ({weekendCount * 100.0 / draws.Count,5:F2}%)");
        Console.WriteLine();

        // Sample draws
        Console.WriteLine("📋 SAMPLE DRAWS:");
        for (int i = 0; i < Math.Min(5, draws.Count); i++)
        {
            Console.WriteLine($"  {draws[i]}");
        }
        Console.WriteLine();
    }
}