"""Exercise the live API: python local_api.py [--url http://127.0.0.1:8000]."""

import argparse

import requests

from app.schemas.prediction import EXAMPLE


def run(url):
    url = url.rstrip("/")
    with requests.Session() as session:
        greeting = session.get(f"{url}/", timeout=10)
        print(f"GET Status Code: {greeting.status_code}")
        greeting.raise_for_status()
        print(f"Result: {greeting.json()['message']}")
        response = session.post(f"{url}/predict", json=EXAMPLE, timeout=10)
        print(f"POST Status Code: {response.status_code}")
        response.raise_for_status()
        print(f"Result: {response.json()['prediction']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    run(parser.parse_args().url)
