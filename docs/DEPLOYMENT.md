# Two-laptop deployment

```
Laptop 2: Streamlit UI (port 8501)  --HTTP-->  Laptop 1: FastAPI + model + decoder (port 8000)
          PATHSNAP_API_BASE_URL=http://<Laptop-1-Wi-Fi-IP>:8000
```

Both laptops on the same Wi-Fi. All commands are Windows PowerShell, run
from the repository root.

## What runs where

Laptop 1, `POST /predict` for one trip:

1. Load `.track`, `.nodes`, `.arcs` (the `.route` file is not opened).
2. Build the road graph and candidate arcs (`pathsnap.candidates`).
3. Extract features (`pathsnap.features`).
4. Score candidates with the trained classifier
   (`artifacts/models/logistic_regression.joblib`).
5. Decode the route with HMM/Viterbi (`pathsnap.decoder`, `configs/decoder.yaml`).
6. In student mode only: score the prediction against the reference route
   on the server and return aggregate numbers (pointwise accuracy, route
   arc recall/precision). Reference arcs are never sent to the UI.

Laptop 2 shows the trip list, sends the selected trip ID, and draws the
returned GPS fixes and predicted route on a Folium map.

### Modes

| `PATHSNAP_MODE` | Partition served | Evaluation shown |
|---|---|---|
| `student` | `data/raw/student_test` (10 trips, supplied) | yes, computed on the server |
| `evaluator` (default) | `data/raw/evaluator_test` (not supplied yet) | no |

Until the evaluator trips are supplied, run with `student`. In evaluator
mode without that folder, `/trips` returns 503.

## API

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status":"ok","mode":...,"partition":...,"model":...,"model_loaded":...}` |
| `GET /trips` | `{"mode":..., "trips":[...]}`; 503 if the partition folder is missing |
| `POST /predict` `{"trip_id": "..."}` | decoded route (below); 404 unknown trip, 422 empty ID, 503 model file missing |

`/predict` response fields: `trip_id`, `status` (`success`,
`success_with_unresolved`, `split`, `failed`), `observation_indices`,
`matched_arc_ids`, `observation_status`, `expanded_route_arc_ids`,
`route_segments`, `route_geometry` (list of segments of `[lat, lon]`),
`gap_boundaries`, `diagnostics`, `runtime_seconds`, `observations` (GPS
fixes as `[lat, lon]`), `timings` (seconds per stage), `evaluation`
(student mode only, otherwise `null`).

Interactive docs: `http://<Laptop-1-Wi-Fi-IP>:8000/docs`.

## Laptop 1: backend

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Data: put the raw trips in `data\raw\student_test\<id>\<id>.{track,nodes,arcs,route}`
(and `data\raw\evaluator_test\...` when supplied).

The trained model is committed at `artifacts/models/logistic_regression.joblib`.
To rebuild it from the training partition (about 10 to 15 minutes, mostly
feature extraction):

```powershell
python scripts/train_final_model.py
```

Start the API:

```powershell
$env:PATHSNAP_MODE = "student"
python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required, otherwise other machines cannot connect. The
window logs every request with the client IP, the trip, the time per stage
and the evaluation. Use this window for the API screenshot in the report.

Find the Wi-Fi IPv4 address (second window):

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Format-Table InterfaceAlias, IPAddress
```

Use the `Wi-Fi` adapter's address. Ignore `127.0.0.1`, `169.254.*` and
virtual adapters (`VMware`, `VirtualBox`, `vEthernet`): they look like LAN
addresses but are not reachable from Laptop 2.

Check:

```powershell
Invoke-RestMethod http://<Laptop-1-Wi-Fi-IP>:8000/health
```

Firewall: when uvicorn first starts, allow Python in the Windows prompt. If
Laptop 2 still cannot connect, open the port (PowerShell as Administrator):

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
```

## Laptop 2: UI

```powershell
python -m pip install -r requirements.txt
Test-NetConnection <Laptop-1-Wi-Fi-IP> -Port 8000
$env:PATHSNAP_API_BASE_URL = "http://<Laptop-1-Wi-Fi-IP>:8000"
streamlit run ui/app.py
```

Run `streamlit` from the repository root so `.streamlit/config.toml` (light
theme) is picked up. `$env:` only lasts for that window.

The sidebar shows the backend URL, **Connected**, the mode and the model.
Pick a trip, press **Predict route**. A prediction takes about 5 to 15
seconds per trip on a laptop CPU.

## Integration checklist

| # | Test | Expected |
|---|---|---|
| 1 | `GET /health` on Laptop 1 | `status: ok`, correct mode |
| 2 | `Test-NetConnection` and UI sidebar on Laptop 2 | `TcpTestSucceeded: True`, Connected |
| 3 | Trip list | 10 student_test IDs in student mode |
| 4 | Predict each trip | HTTP 200, map with GPS and predicted route, metrics shown |
| 5 | Laptop 1 terminal | one log block per request with Laptop 2's IP |
| 6 | Response contains no reference route | only the aggregate `evaluation` |
| 7 | Evaluator mode without data | sidebar connected, trip list shows the 503 message |
| 8 | Stop the backend | UI shows Disconnected and the URL it tried |
| 9 | Wrong IP in `PATHSNAP_API_BASE_URL` | Disconnected after about 4 s |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Connection refused | Backend not running or started without `--host 0.0.0.0` |
| Timeout / wrong IP | Use the Wi-Fi adapter IP; same Wi-Fi on both; `Test-NetConnection` |
| Works on Laptop 1 only | Firewall rule above; some campus/guest Wi-Fi blocks device-to-device traffic, use a phone hotspot |
| Port 8000 in use | `netstat -ano \| findstr :8000`, then `Stop-Process -Id <PID>`, or use another port everywhere |
| `No module named 'pathsnap'` | `python -m pip install -e .` |
| `No module named ...` (fastapi, streamlit, lightgbm ...) | `python -m pip install -r requirements.txt` |
| 503 "Trip data is unavailable" | Wrong mode or data not under `data\raw\<partition>`; set `$env:PATHSNAP_MODE` / `$env:PATHSNAP_DATA_ROOT` and restart |
| 503 "Trained model not found" | `python scripts/train_final_model.py` |
| 404 on `/predict` | Trip not in the served partition; press Recheck to reload the list |
| 500 | Full traceback is in the Laptop 1 terminal |
| `streamlit` not recognised | `python -m streamlit run ui/app.py` |
