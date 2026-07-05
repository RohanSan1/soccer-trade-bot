#!/usr/bin/env python3
"""Hard-stop watchdog for OVH AI Training jobs.

Runs locally on the Mac. Monitors OVH job status via CDP (kimi-webbridge)
and kills any running GPU job before the hard time limit.

Usage:
    python infra/hard_stop.py --job-name ensemble-final --max-hours 6.5

Timeline:
    T+0:00  Job submitted
    T+max-0.5h  Kill any running job (30min buffer)
    T+max  Final kill + exit
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ID = "b388425bd0bd4fccb9be4aefeaa77eb9"
CDP_URL = "http://127.0.0.1:10086/command"


def cdp_eval(expression: str, timeout: int = 30) -> str:
    """Evaluate JS expression via kimi-webbridge CDP."""
    payload = json.dumps({
        "action": "evaluate",
        "args": {"code": expression}
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), "-X", "POST", CDP_URL,
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        data = json.loads(result.stdout)
        return data.get("data", {}).get("value", "")
    except Exception as e:
        return f"err:{e}"


def get_job_state(job_name: str) -> str:
    """Get the state of a job by name."""
    js = f"""(() => {{
        return new Promise((resolve) => {{
            const iframe = document.querySelector("iframe[src]");
            const win = iframe.contentWindow;
            win.fetch("/engine/apiv6/cloud/project/{PROJECT_ID}/ai/job").then(r=>r.json()).then(j => {{
                const job = j.find(x => x.spec.name === "{job_name}" && ["RUNNING","PENDING","QUEUED","FINALIZING"].includes(x.status.state));
                resolve(job ? JSON.stringify({{id: job.id, state: job.status.state, name: job.spec.name}}) : "none");
            }}).catch(e => resolve("err:" + e.message));
        }});
    }})()"""
    return cdp_eval(js)


def kill_job(job_id: str) -> str:
    """Kill a running OVH AI Training job."""
    js = f"""(() => {{
        return new Promise((resolve) => {{
            const iframe = document.querySelector("iframe[src]");
            const win = iframe.contentWindow;
            win.fetch("/engine/apiv6/cloud/project/{PROJECT_ID}/ai/job/{job_id}/kill", {{
                method: "PUT",
                credentials: "include"
            }}).then(r => resolve("killed:" + r.status)).catch(e => resolve("err:" + e.message));
        }});
    }})()"""
    return cdp_eval(js)


def watchdog_loop(job_name: str, max_hours: float, poll_interval: int = 60):
    """Main watchdog loop. Kills job at max_hours."""
    start_time = time.time()
    kill_at = start_time + (max_hours - 0.5) * 3600  # 30min buffer
    hard_stop_at = start_time + max_hours * 3600

    logger.info("=== HARD STOP WATCHDOG ===")
    logger.info("Job: %s", job_name)
    logger.info("Max hours: %.1f", max_hours)
    logger.info("Kill at: %s (T+%.1fh)", datetime.fromtimestamp(kill_at).strftime("%H:%M:%S"), max_hours - 0.5)
    logger.info("Hard stop: %s (T+%.1fh)", datetime.fromtimestamp(hard_stop_at).strftime("%H:%M:%S"), max_hours)
    logger.info("Poll interval: %ds", poll_interval)
    logger.info("")

    killed = False
    while True:
        now = time.time()
        elapsed = now - start_time
        remaining = hard_stop_at - now

        if remaining <= 0:
            logger.info("HARD STOP REACHED. Force killing any remaining jobs.")
            state = get_job_state(job_name)
            if state != "none":
                try:
                    job_info = json.loads(state)
                    kill_job(job_info["id"])
                    logger.info("KILLED %s", job_info["id"])
                except Exception:
                    pass
            logger.info("Watchdog exiting. All artifacts should be in Object Storage.")
            sys.exit(0)

        # Check job status
        state = get_job_state(job_name)

        if state == "none":
            logger.info("[T+%.1fh] Job not running (DONE/FAILED/not found). Elapsed: %.0fs remaining: %.0fs",
                       elapsed / 3600, elapsed, remaining)
            if elapsed > 300:  # If we've been running >5min and job is done, exit
                logger.info("Job appears complete. Watchdog exiting.")
                sys.exit(0)
        else:
            try:
                job_info = json.loads(state)
                logger.info("[T+%.1fh] Job %s: %s | Remaining: %.0fs",
                           elapsed / 3600, job_info["name"], job_info["state"], remaining)
            except Exception:
                logger.info("[T+%.1fh] Job state: %s | Remaining: %.0fs", elapsed / 3600, state, remaining)

        # Kill at buffer time
        if not killed and now >= kill_at:
            logger.info("")
            logger.info("*** KILL TIME REACHED (T+%.1fh) ***", max_hours - 0.5)
            if state != "none":
                try:
                    job_info = json.loads(state)
                    result = kill_job(job_info["id"])
                    logger.info("Kill result: %s", result)
                    killed = True
                except Exception as e:
                    logger.error("Failed to kill: %s", e)
            else:
                logger.info("No job to kill.")
                killed = True

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Hard-stop watchdog for OVH AI Training")
    parser.add_argument("--job-name", required=True, help="OVH job name to monitor")
    parser.add_argument("--max-hours", type=float, default=6.5, help="Max hours before hard stop")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    watchdog_loop(args.job_name, args.max_hours, args.poll_interval)


if __name__ == "__main__":
    main()
