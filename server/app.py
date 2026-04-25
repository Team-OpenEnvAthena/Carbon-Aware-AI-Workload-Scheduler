"""FastAPI app for CarbonSchedulerEnv using OpenEnv create_app()."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: uv sync") from e

try:
    from ..models import CarbonSchedulerAction, CarbonSchedulerObservation
    from .CarbonSchedulerEnv_environment import CarbonSchedulerEnvEnvironment
except ModuleNotFoundError:
    from models import CarbonSchedulerAction, CarbonSchedulerObservation
    from server.CarbonSchedulerEnv_environment import CarbonSchedulerEnvEnvironment

app = create_app(
    CarbonSchedulerEnvEnvironment,
    CarbonSchedulerAction,
    CarbonSchedulerObservation,
    env_name="CarbonSchedulerEnv",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
