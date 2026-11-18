"""Service-layer integration point for future PathSnap inference.

Keep API route handlers in ``main.py`` thin. Once the data and inference
interfaces are agreed, place orchestration behind a service in this module and
wire it into the application through FastAPI dependencies or lifespan state.

No prediction behavior is defined here yet because model and input contracts
are still being developed by the team.
"""
