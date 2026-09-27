"""HTTP API for the IWNET grape leaf disease classifier."""

__all__ = ["app"]


def __getattr__(name: str):
    # Lazy so that `import iwnet.api` does not pull in torch/FastAPI eagerly.
    if name == "app":
        from iwnet.api.app import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
