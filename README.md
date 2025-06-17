# Cyclic Voltammetry Simulator

This repository contains a minimal web service that simulates cyclic voltammetry using a Python backend. The simulation uses only Python's standard library.

## Running the server

```bash
python server.py
```

The server listens on `http://localhost:8000`. Open this address in a web browser to load `index.html`. Fill out the form parameters and run the simulation. The resulting current vs. potential curve is displayed using Chart.js.

## Files

- `simulation.py` – Implements the cyclic voltammetry simulation.
- `server.py` – Simple HTTP server that exposes a `/simulate` endpoint and serves `index.html`.
- `index.html` – Web interface for entering parameters and viewing the graph.
