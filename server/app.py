"""FastAPI app for CarbonSchedulerEnv using OpenEnv create_app()."""

# Why absolute imports only:
# uvicorn runs this as `server.app:app` from /app/env.
# `server` is therefore the top-level package — there is no parent package
# above it. Relative imports like `from ..models` would try to go above
# the top-level package, which Python forbids with:
#   "attempted relative import beyond top-level package"
#
# PYTHONPATH=/app/env is set in the Dockerfile, so both `models` and
# `server.*` are always importable as absolute paths.

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. Run: uv sync") from e

from models import CarbonSchedulerAction, CarbonSchedulerObservation
from server.CarbonSchedulerEnv_environment import CarbonSchedulerEnvEnvironment

app = create_app(
    CarbonSchedulerEnvEnvironment,
    CarbonSchedulerAction,
    CarbonSchedulerObservation,
    env_name="CarbonSchedulerEnv",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)