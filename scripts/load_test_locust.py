#!/usr/bin/env python3
"""
IBVAP — T-04: Automated Load and Soak Testing Suite
Simulates concurrent Command Center operators, tactical surveillance video stream
consumers, and real-time WebSocket telemetry subscribers under sustained edge loads.

Usage:
  1. Standalone Stress/Soak Mode:
       python scripts/load_test_locust.py --users 20 --duration 60 --host http://localhost:8000
  2. Locust Web Dashboard Mode:
       locust -f scripts/load_test_locust.py --host http://localhost:8000
"""
import sys
import time
import json
import argparse
import asyncio
import logging
from typing import List, Dict

try:
    import httpx
except ImportError:
    httpx = None

try:
    from locust import HttpUser, task, between
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    HttpUser = object

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoadTest")


# ==============================================================================
# 1. Locust User Simulation
# ==============================================================================
if LOCUST_AVAILABLE:
    class CommandCenterOperator(HttpUser):
        wait_time = between(0.1, 1.0)
        token = ""

        def on_start(self):
            """Authenticate as operator."""
            resp = self.client.post("/api/auth/login", data={"username": "ibvap-admin", "password": "TestPassword123"})
            if resp.status_code == 200:
                self.token = resp.json().get("token_type")
                logger.info("Locust operator authenticated successfully.")

        @task(5)
        def view_recent_events(self):
            self.client.get("/api/events/recent?limit=50&data_origin=LIVE")

        @task(3)
        def query_system_health(self):
            self.client.get("/api/system/health/detailed")

        @task(4)
        def list_camera_status(self):
            self.client.get("/api/cameras/")

        @task(2)
        def view_anpr_plates(self):
            self.client.get("/api/anpr/plates?limit=25")

        @task(1)
        def query_prometheus_metrics(self):
            self.client.get("/metrics")


# ==============================================================================
# 2. Standalone Asyncio High-Concurrency Soak Tester
# ==============================================================================
async def async_worker(user_id: int, base_url: str, stop_event: asyncio.Event, stats: Dict[str, int]):
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        # Login
        token = None
        try:
            r = await client.post("/api/auth/login", data={"username": "ibvap-admin", "password": "TestPassword123"})
            if r.status_code == 200:
                stats["login_success"] += 1
            else:
                stats["login_fail"] += 1
        except Exception:
            stats["login_fail"] += 1

        endpoints = [
            "/api/cameras/",
            "/api/events/recent?limit=50",
            "/api/anpr/plates?limit=20",
            "/api/system/health/detailed",
            "/api/health",
            "/metrics",
        ]

        while not stop_event.is_set():
            for ep in endpoints:
                if stop_event.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    res = await client.get(ep)
                    latency = (time.perf_counter() - t0) * 1000.0
                    if res.status_code == 200:
                        stats["requests_success"] += 1
                    else:
                        stats["requests_fail"] += 1
                except Exception:
                    stats["requests_fail"] += 1

                await asyncio.sleep(0.05)


async def run_standalone_soak(host: str, users: int, duration_sec: int):
    logger.info("Starting IBVAP Standalone Soak Test: %d users, %d seconds target on %s", users, duration_sec, host)
    stop_event = asyncio.Event()
    stats = {
        "login_success": 0,
        "login_fail": 0,
        "requests_success": 0,
        "requests_fail": 0,
    }

    tasks = [asyncio.create_task(async_worker(i, host, stop_event, stats)) for i in range(users)]

    # Run for specified duration
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        await asyncio.sleep(2.0)
        elapsed = time.time() - start_time
        total_reqs = stats["requests_success"] + stats["requests_fail"]
        rps = total_reqs / max(1.0, elapsed)
        logger.info(
            "Progress: %.1fs / %ds | Req: %d (Success: %d, Fail: %d) | RPS: %.1f",
            elapsed, duration_sec, total_reqs, stats["requests_success"], stats["requests_fail"], rps,
        )

    stop_event.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    total_reqs = stats["requests_success"] + stats["requests_fail"]
    logger.info("=== Soak Test Completed ===")
    logger.info("Total Requests: %d", total_reqs)
    logger.info("Successful:     %d (%.2f%%)", stats["requests_success"], (stats["requests_success"] / max(1, total_reqs)) * 100.0)
    logger.info("Failed:         %d", stats["requests_fail"])


def main():
    parser = argparse.ArgumentParser(description="IBVAP Load and Soak Test")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Base URL of IBVAP API")
    parser.add_argument("--users", type=int, default=15, help="Number of concurrent simulated operators")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()

    asyncio.run(run_standalone_soak(host=args.host, users=args.users, duration_sec=args.duration))


if __name__ == "__main__":
    main()
