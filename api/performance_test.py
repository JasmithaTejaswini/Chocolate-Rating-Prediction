import time
import statistics
import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8001/predict"

# Test input
sample = {
    "company": "A. Morin",
    "specific_bean_origin_or_bar_name": "Chuao",
    "review_date": 2016,
    "cocoa_percent": 70,
    "company_location": "France",
    "bean_type": "Trinitario",
    "broad_bean_origin": "Venezuela"
}

TOTAL_REQUESTS = 50

latencies = []
predictions = []
successful = 0
failed = 0

print("API PERFORMANCE TESTING")
print("--------------------------------")

start_total = time.perf_counter()

for i in range(TOTAL_REQUESTS):

    start = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json=sample,
            timeout=10
        )

        latency = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            successful += 1
            latencies.append(latency)

            result = response.json()
            predictions.append(result["predicted_rating"])

        else:
            failed += 1

    except requests.RequestException:
        failed += 1

total_time = time.perf_counter() - start_total

if latencies:

    report = {
        "total_requests": TOTAL_REQUESTS,
        "successful_requests": successful,
        "failed_requests": failed,
        "average_latency_ms": round(statistics.mean(latencies), 3),
        "minimum_latency_ms": round(min(latencies), 3),
        "maximum_latency_ms": round(max(latencies), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "throughput_requests_per_second": round(
            successful / total_time, 3
        ),
        "prediction_consistent": len(set(predictions)) == 1
    }

    output_path = Path("artifacts/api_performance_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(json.dumps(report, indent=4))
    print("\nPerformance report saved:", output_path)

else:
    print("All API requests failed. Check whether the API is running.")