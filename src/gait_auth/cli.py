"""Command-line entry points for the prototype."""
from __future__ import annotations
import argparse
from pathlib import Path
from .risk_engine import classify_transaction_risk, decide
from .experiments import status_e2, status_e3, status_e4, status_e6, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gait-based mobile-money authentication prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    risk = sub.add_parser("decision", help="evaluate the risk-adaptive policy")
    risk.add_argument("--confidence", choices=["high", "medium", "low"], required=True)
    risk.add_argument("--amount", type=float, required=True)
    risk.add_argument("--new-recipient", action="store_true")
    risk.add_argument("--anomalous-device-location", action="store_true")
    statuses = sub.add_parser("experiment-status", help="write honest availability statuses for E2-E6")
    statuses.add_argument("--output", default="results/json/experiment_status.json")
    args = parser.parse_args(argv)
    if args.command == "decision":
        risk_level = classify_transaction_risk(args.amount, args.new_recipient, args.anomalous_device_location)
        print({"transaction_risk": risk_level.value, "decision": decide(args.confidence, risk_level).value})
    elif args.command == "experiment-status":
        payload = [s.to_dict() for s in [status_e2(set()), status_e3(set()), status_e4(False), status_e6(False)]]
        save_json(payload, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
