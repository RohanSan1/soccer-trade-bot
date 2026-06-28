# Soccer Vision-to-Trade Bot

Autonomous soccer livestream analysis and prediction market trading system.

**All compute runs on cloud infrastructure** — OVH H100 for training, Lightning AI for inference. Nothing runs locally.

## Architecture

```
GitHub Actions → OVH H100 (training) → Lightning AI (inference) → Polymarket/Kalshi
```

### Pipeline Flow

1. **Frame Extraction**: FFmpeg captures 1fps from HLS/RTMP stream
2. **OCR**: PaddleOCR extracts score, clock, team names (99.6% accuracy)
3. **Event Detection**: CLIP ViT-L/14 classifies goals, red cards, VAR reviews
4. **Player Detection**: YOLOv10-X computes pressure zone signal
5. **Prediction**: XGBoost + LightGBM ensemble → calibrated win probabilities
6. **Trading**: Edge detection → Quarter-Kelly sizing → Order placement

## Quick Start

### 1. Setup GitHub Secrets

| Secret | Description |
|--------|-------------|
| `OVH_APP_KEY` | OVH API application key |
| `OVH_APP_SECRET` | OVH API application secret |
| `OVH_CONSUMER_KEY` | OVH API consumer key |
| `OVH_PROJECT_ID` | OVH Public Cloud project ID |
| `OVH_SSH_PRIVATE_KEY` | SSH private key for OVH instance |
| `LIGHTNING_USER_ID` | Lightning AI user ID |
| `LIGHTNING_API_KEY` | Lightning AI API key |
| `POLYMARKET_PRIVATE_KEY` | Wallet private key for Polymarket CLOB |
| `KALSHI_API_KEY` | Kalshi API key ID |
| `KALSHI_PRIVATE_KEY` | Kalshi RSA private key (PEM format) |

### 2. Train the Model

Trigger `train_model.yml` from GitHub Actions:

```
gh workflow run train_model.yml \
  -f epochs=100 \
  -f grid_search=false
```

This will:
- Provision OVH H100 instance (~$3.60/hr)
- Build dataset from 8 sources (~2.1M snapshots)
- Train XGBoost + LightGBM ensemble
- Fine-tune YOLOv10-X (100 epochs)
- Fine-tune CLIP ViT-L/14
- Upload model artifacts to Lightning AI Drive
- Terminate OVH instance

**Estimated cost: ~$31.50 for ~8.75 hours**

### 3. Paper Trade

Before live trading, run in paper mode:

```
gh workflow run paper_trade.yml \
  -f stream_url="https://example.com/match.m3u8" \
  -f match_id="match_001"
```

All signals are logged to SQLite. No orders are placed.

### 4. Live Trade

Once paper trading validates:

```
gh workflow run deploy_stream_worker.yml \
  -f stream_url="https://example.com/match.m3u8" \
  -f match_id="match_001" \
  -f dry_run=false
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_URL` | - | HLS/RTMP stream URL |
| `MATCH_ID` | - | Match identifier |
| `DRY_RUN` | `true` | Paper trade mode |
| `MIN_BET_USD` | `5.0` | Minimum bet size |
| `MAX_BET_PCT` | `0.02` | Max bet as % of bankroll |
| `KELLY_FRACTION` | `0.25` | Kelly fraction |
| `EDGE_THRESHOLD` | `0.05` | Minimum edge to trade |
| `OCR_CONFIDENCE_THRESHOLD` | `0.70` | OCR confidence cutoff |
| `STREAM_LAG_MAX` | `8` | Max stream lag (seconds) |
| `SCORE_ROI` | - | Score bounding box (x1,y1,x2,y2) |
| `CLOCK_ROI` | - | Clock bounding box (x1,y1,x2,y2) |

## Model Training

### Dataset (2.1M snapshots)

| Source | Snapshots | Features |
|--------|-----------|----------|
| StatsBomb Open Data | ~50K | Events, xG, shots |
| Understat | ~550K | xG, shot locations |
| SoccerNet Events | ~27K | Action labels |
| WyScout | ~107K | Player tracking |
| European Soccer DB | ~1.37M | Match results |
| FBref | Enrichment | Form, PPDA, xG |
| Transfermarkt | Enrichment | Squad values, injuries |
| Club ELO | Enrichment | Historical ratings |

### Features (38 total)

- **Live state**: score_diff, clock, red_cards, pressure, xG, shots
- **Interaction**: `score_diff × time_remaining` (most predictive)
- **Pre-match**: ELO, form, H2H, squad value, injuries
- **Tactical**: pressing intensity, xG form, xG conceded
- **Context**: competition tier, importance, fatigue
- **Momentum**: recent goals, cards, xG delta

### Training Config

- **XGBoost**: 1500 estimators, max_depth=6, lr=0.05
- **LightGBM**: 1500 estimators, num_leaves=63
- **Ensemble**: Weighted average (50/50) + isotonic calibration
- **Validation**: GroupKFold by match_id (no data leakage)

## Kill Switch

Trading halts automatically if:

1. Stream lag > 8 seconds
2. OCR confidence < 0.70 for 5 consecutive reads
3. API errors > 3 in last 60 seconds
4. Bankroll drawdown > 20%
5. Unhandled exception in signal loop

## Project Structure

```
soccer-trade-bot/
├── .github/workflows/     # CI/CD orchestration
├── vision/                # Frame extraction, OCR, CLIP, YOLO
├── model/                 # XGBoost, LightGBM, calibration
├── market/                # Polymarket, Kalshi integration
├── trading/               # Edge calc, Kelly sizing, signal engine
├── data/                  # Dataset building, logging
├── infra/                 # OVH, Lightning AI provisioning
├── config.py              # All configuration
└── main.py                # Entrypoint
```

## License

MIT
