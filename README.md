# NeoNet AI Miner

Connect your computer to the NeoNet network and earn NEO tokens by providing AI computing power.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Miner

```bash
python neonet_miner.py --server https://YOUR_NEONET_SERVER
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--server` | NeoNet server URL | http://localhost:8000 |
| `--id` | Your miner ID (auto-generated if not set) | random |
| `--cpu` | Number of CPU cores to use | 4 |
| `--gpu-mem` | GPU memory in MB (0 = no GPU) | 0 |
| `--gpu-model` | GPU model name | "" |

### Examples

**Basic CPU Mining:**
```bash
python neonet_miner.py --server https://neonet.example.com --cpu 8
```

**GPU Mining (NVIDIA):**
```bash
python neonet_miner.py --server https://neonet.example.com --gpu-mem 8192 --gpu-model "RTX 4090"
```

**Custom Miner ID:**
```bash
python neonet_miner.py --server https://neonet.example.com --id my_miner_001
```

## How It Works

1. **Register** - Your miner connects to NeoNet and registers its hardware specs
2. **Start Session** - A mining session begins and tasks are assigned
3. **Process Tasks** - Your computer processes real AI workloads:
   - Matrix multiplication for neural networks
   - Gradient computation for model training
   - Model inference for fraud detection
   - Data hashing for blockchain verification
   - Tensor reduction for model aggregation
4. **Earn Rewards** - Each completed task earns you NEO tokens

## Task Types & Rewards

| Task Type | Description | Reward |
|-----------|-------------|--------|
| `matrix_multiply` | Neural network layer computation | 0.05 NEO |
| `gradient_compute` | Model training gradients | 0.08 NEO |
| `model_inference` | Run fraud detection model | 0.06 NEO |
| `data_hash` | Hash blockchain data | 0.03 NEO |
| `tensor_reduce` | Aggregate model data | 0.04 NEO |

## Sample Output

```
============================================================
    NeoNet AI Miner - Proof of Intelligence Network
============================================================
Server: https://neonet.example.com
Miner ID: miner_a1b2c3d4e5f6
CPU Cores: 8
GPU: RTX 4090 (8192MB)
------------------------------------------------------------
[OK] Registered with NeoNet: {'success': True, ...}
[OK] Session started: 550e8400-e29b-41d4-a716-446655440000

[MINING] Starting AI task processing...
[INFO] Press Ctrl+C to stop

[TASK] Processing: gradient_compute (ID: 7a8b9c0d...)
[DONE] Computed in 156ms
[REWARD] +0.0800 NEO | Total: 0.0800 NEO

[TASK] Processing: matrix_multiply (ID: 1e2f3a4b...)
[DONE] Computed in 89ms
[REWARD] +0.0500 NEO | Total: 0.1300 NEO

[STATUS] Tasks: 2 | Rewards: 0.1300 NEO
```

## API Endpoints Used

The miner client communicates with these NeoNet endpoints:

- `POST /ai-energy/register` - Register as contributor
- `POST /ai-energy/start-session` - Start mining session
- `GET /ai-energy/task/{contributor_id}` - Fetch task to process
- `POST /ai-energy/submit-result` - Submit completed task
- `POST /ai-energy/heartbeat` - Keep session alive
- `POST /ai-energy/stop-session` - End session

## Running as a Service (Linux)

Create a systemd service file `/etc/systemd/system/neonet-miner.service`:

```ini
[Unit]
Description=NeoNet AI Miner
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/miner-client
ExecStart=/usr/bin/python3 neonet_miner.py --server https://neonet.example.com --cpu 8
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl enable neonet-miner
sudo systemctl start neonet-miner
```

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY neonet_miner.py .
CMD ["python", "neonet_miner.py", "--server", "https://neonet.example.com"]
```

Build and run:

```bash
docker build -t neonet-miner .
docker run -d neonet-miner
```

## Troubleshooting

**Cannot connect to server:**
- Check that the server URL is correct
- Ensure your firewall allows outbound connections

**Session keeps dropping:**
- Unstable internet connection
- Server may be overloaded, try again later

**Low rewards:**
- Consider using a GPU for faster processing
- Increase CPU cores

## Support

- GitHub: https://github.com/NeoNet-Blockchain/miner-client
- Discord: https://discord.gg/neonet
- Docs: https://docs.neonet.network
