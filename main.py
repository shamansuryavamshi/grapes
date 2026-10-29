"""Vercel entrypoint.

Vercel only auto-detects Python apps at the project root or under src/, app/ or
api/. The FastAPI instance lives in iwnet/api/app.py, so re-export it here.
"""

from iwnet.api.app import app

__all__ = ["app"]
