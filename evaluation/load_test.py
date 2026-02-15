import requests
import time
import concurrent.futures

URL = "http://localhost:8000/chat"
PAYLOAD = {"query": "What encryption standard is required?"}


def send_request():
    start = time.time()
    r = requests.post(URL, json=PAYLOAD)
    return time.time() - start


def run_load_test(num_requests=50):

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        latencies = [f.result() for f in futures]

    print("Avg Latency:", sum(latencies) / len(latencies))
    print("Max Latency:", max(latencies))


if __name__ == "__main__":
    run_load_test()
