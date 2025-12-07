import requests
from bs4 import BeautifulSoup
import csv

YEARS = [2024, 2025]

def fetch_year(year: int):
    url = f"https://www.lottery.net/north-carolina/cash-pop/numbers/{year}"
    print(f"Fetching {url}...")

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    table = soup.find("table", class_="prizes")

    if not table:
        print("No table found!")
        return rows

    for tr in table.find_all("tr")[1:]:  # skip header row
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # Extract datetime cell
        datetime_html = tds[0].decode_contents().strip()
        parts = datetime_html.split("<br>")

        # ---- Parse day & time ----
        day_of_week = "Unknown"
        draw_time = "Unknown"
        date_text = "Unknown"

        if len(parts) == 2:
            # Normal case: "Tuesday - 11:59pm" + "December 31, 2024"
            day_time_raw = BeautifulSoup(parts[0], "html.parser").get_text(strip=True)
            date_text = BeautifulSoup(parts[1], "html.parser").get_text(strip=True)

            if " - " in day_time_raw:
                day_of_week, draw_time = [x.strip() for x in day_time_raw.split(" - ")]
            else:
                day_of_week = day_time_raw

        elif len(parts) == 1:
            # Only a single line (rare)
            line = BeautifulSoup(parts[0], "html.parser").get_text(strip=True)

            # Try to auto-detect format
            if " " in line:
                date_text = line
            else:
                day_of_week = line

        # ---- Draw number ----
        draw_number_raw = tds[1].get_text(strip=True)
        try:
            draw_number = int(draw_number_raw)
        except:
            draw_number = None

        # ---- Result number ----
        li = tds[2].find("li", class_="ball")
        result = None
        if li:
            try:
                result = int(li.get_text(strip=True))
            except:
                pass

        rows.append({
            "year": year,
            "date": date_text,
            "day_of_week": day_of_week,
            "draw_time": draw_time,
            "draw_number": draw_number,
            "result": result
        })

    return rows


# Collect all draws
all_rows = []
for y in YEARS:
    all_rows.extend(fetch_year(y))

print(f"\nTotal draws collected: {len(all_rows)}")

# Save CSV
output = "nc_cash_pop_full_metadata.csv"
with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["year", "date", "day_of_week", "draw_time", "draw_number", "result"]
    )
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved to {output}")
