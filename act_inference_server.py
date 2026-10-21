#!/usr/bin/env python3
"""Local Unix-socket LeRobot policy sidecar for Isaac Sim's Python runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import traceback

from act_runtime import ACTRuntimePolicy, decode_rgb_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--inference-seed", type=int, default=20261018)
    args = parser.parse_args()

    policy = ACTRuntimePolicy(
        args.checkpoint,
        device=args.device,
        action_steps=args.action_steps,
        inference_seed=args.inference_seed,
    )
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    print(f"ACT_SIDECAR_READY {args.socket}", flush=True)
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb") as stream:
            while line := stream.readline():
                operation = None
                try:
                    request = json.loads(line)
                    operation = request.get("op")
                    if operation == "close":
                        response = {"ok": True}
                    elif operation == "describe":
                        response = {
                            "ok": True,
                            "policy_type": policy.policy_type,
                            "language_conditioned": policy.language_conditioned,
                            "state_dim": int(
                                policy.policy.config.input_features["observation.state"].shape[0]
                            ),
                            "action_steps": policy.action_steps,
                        }
                    elif operation == "reset":
                        policy.policy.reset()
                        response = {"ok": True}
                    elif operation == "infer":
                        actions = policy.predict_chunk(
                            decode_rgb_png(request["overhead_png"]),
                            decode_rgb_png(request["wrist_png"]),
                            request["state"],
                            request.get("instruction"),
                        )
                        response = {"ok": True, "actions": actions.tolist()}
                    else:
                        raise ValueError(f"Unknown operation: {operation!r}")
                except Exception:
                    response = {"ok": False, "error": traceback.format_exc()[-4000:]}
                stream.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
                stream.flush()
                if operation == "close":
                    break
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
