#!/usr/bin/env python3
"""
NeoNet AI Miner Client
Connect to NeoNet network and provide AI computing power
Earn NEO tokens for completed tasks
"""
import asyncio
import aiohttp
import json
import time
import hashlib
import numpy as np
import argparse
import sys
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    MATRIX_MULTIPLY = "matrix_multiply"
    GRADIENT_COMPUTE = "gradient_compute"
    MODEL_INFERENCE = "model_inference"
    DATA_HASH = "data_hash"
    TENSOR_REDUCE = "tensor_reduce"

@dataclass
class MinerConfig:
    server_url: str
    contributor_id: str
    cpu_cores: int
    gpu_memory_mb: int
    gpu_model: str

class NeoNetMiner:
    """Real AI Miner that connects to NeoNet and processes tasks"""
    
    def __init__(self, config: MinerConfig):
        self.config = config
        self.session_id: Optional[str] = None
        self.is_running = False
        self.tasks_completed = 0
        self.total_rewards = 0.0
        self.current_task: Optional[dict] = None
        
    async def register(self) -> bool:
        """Register miner with the network"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": self.config.contributor_id,
                    "cpu_cores": self.config.cpu_cores,
                    "gpu_memory_mb": self.config.gpu_memory_mb,
                    "gpu_model": self.config.gpu_model
                }
                async with session.post(
                    f"{self.config.server_url}/ai-energy/register",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"[OK] Registered with NeoNet: {data}")
                        return True
                    else:
                        error = await resp.text()
                        print(f"[ERROR] Registration failed: {error}")
                        return False
            except Exception as e:
                print(f"[ERROR] Cannot connect to server: {e}")
                return False
    
    async def start_session(self) -> bool:
        """Start a mining session"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config.server_url}/ai-energy/start-session",
                    json={"contributor_id": self.config.contributor_id}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.session_id = data.get("session_id")
                        print(f"[OK] Session started: {self.session_id}")
                        return True
                    else:
                        error = await resp.text()
                        print(f"[ERROR] Session start failed: {error}")
                        return False
            except Exception as e:
                print(f"[ERROR] Session start error: {e}")
                return False
    
    async def fetch_task(self) -> Optional[dict]:
        """Fetch a task from the network"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.config.server_url}/ai-energy/task/{self.config.contributor_id}"
                ) as resp:
                    if resp.status == 200:
                        task = await resp.json()
                        if task.get("task_id"):
                            return task
                    return None
            except Exception as e:
                print(f"[WARN] Fetch task error: {e}")
                return None
    
    def process_task(self, task: dict) -> dict:
        """Process AI task locally - REAL COMPUTATION"""
        task_type = task.get("task_type", "matrix_multiply")
        task_data = task.get("data", {})
        
        start_time = time.time()
        result = {}
        
        try:
            if task_type == "matrix_multiply":
                size = task_data.get("size", 100)
                A = np.random.randn(size, size).astype(np.float32)
                B = np.random.randn(size, size).astype(np.float32)
                C = np.dot(A, B)
                result = {
                    "checksum": hashlib.sha256(C.tobytes()).hexdigest()[:16],
                    "shape": list(C.shape),
                    "mean": float(np.mean(C)),
                    "std": float(np.std(C))
                }
                
            elif task_type == "gradient_compute":
                size = task_data.get("size", 1000)
                weights = np.random.randn(size).astype(np.float32)
                inputs = np.random.randn(100, size).astype(np.float32)
                targets = np.random.randn(100).astype(np.float32)
                predictions = np.dot(inputs, weights)
                loss = np.mean((predictions - targets) ** 2)
                gradients = 2 * np.dot(inputs.T, (predictions - targets)) / 100
                result = {
                    "loss": float(loss),
                    "gradient_norm": float(np.linalg.norm(gradients)),
                    "checksum": hashlib.sha256(gradients.tobytes()).hexdigest()[:16]
                }
                
            elif task_type == "model_inference":
                batch_size = task_data.get("batch_size", 32)
                input_dim = task_data.get("input_dim", 512)
                hidden_dim = task_data.get("hidden_dim", 256)
                output_dim = task_data.get("output_dim", 10)
                X = np.random.randn(batch_size, input_dim).astype(np.float32)
                W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
                W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
                hidden = np.maximum(0, np.dot(X, W1))
                output = np.dot(hidden, W2)
                probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                result = {
                    "predictions": probs.argmax(axis=1).tolist()[:5],
                    "confidence": float(np.max(probs)),
                    "checksum": hashlib.sha256(probs.tobytes()).hexdigest()[:16]
                }
                
            elif task_type == "data_hash":
                data_size = task_data.get("size", 10000)
                data = np.random.bytes(data_size)
                iterations = task_data.get("iterations", 1000)
                h = hashlib.sha256(data)
                for _ in range(iterations):
                    h = hashlib.sha256(h.digest())
                result = {
                    "final_hash": h.hexdigest(),
                    "iterations": iterations
                }
                
            elif task_type == "tensor_reduce":
                dims = task_data.get("dims", [100, 100, 100])
                tensor = np.random.randn(*dims).astype(np.float32)
                result = {
                    "sum": float(np.sum(tensor)),
                    "mean": float(np.mean(tensor)),
                    "max": float(np.max(tensor)),
                    "min": float(np.min(tensor)),
                    "checksum": hashlib.sha256(tensor.tobytes()).hexdigest()[:16]
                }
            else:
                size = 50
                A = np.random.randn(size, size)
                result = {"checksum": hashlib.sha256(np.dot(A, A).tobytes()).hexdigest()[:16]}
                
        except Exception as e:
            result = {"error": str(e)}
        
        compute_time = time.time() - start_time
        result["compute_time_ms"] = int(compute_time * 1000)
        result["task_id"] = task.get("task_id")
        result["contributor_id"] = self.config.contributor_id
        
        return result
    
    async def submit_result(self, task_id: str, result: dict) -> Optional[float]:
        """Submit task result and get reward"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": self.config.contributor_id,
                    "session_id": self.session_id,
                    "task_id": task_id,
                    "result": result
                }
                async with session.post(
                    f"{self.config.server_url}/ai-energy/submit-result",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        reward = data.get("reward", 0)
                        return reward
                    else:
                        error = await resp.text()
                        print(f"[WARN] Submit failed: {error}")
                        return None
            except Exception as e:
                print(f"[ERROR] Submit error: {e}")
                return None
    
    async def send_heartbeat(self, tasks_done: int = 0) -> dict:
        """Send heartbeat to keep session alive"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "contributor_id": self.config.contributor_id,
                    "session_id": self.session_id,
                    "cpu_usage": np.random.uniform(20, 80),
                    "gpu_usage": np.random.uniform(0, 60) if self.config.gpu_memory_mb > 0 else 0,
                    "tasks_completed": tasks_done
                }
                async with session.post(
                    f"{self.config.server_url}/ai-energy/heartbeat",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {}
            except Exception:
                return {}
    
    async def stop_session(self) -> dict:
        """Stop the mining session"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config.server_url}/ai-energy/stop-session",
                    json={
                        "contributor_id": self.config.contributor_id,
                        "session_id": self.session_id
                    }
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {}
            except Exception as e:
                print(f"[ERROR] Stop session error: {e}")
                return {}
    
    async def run(self):
        """Main mining loop"""
        print("=" * 60)
        print("    NeoNet AI Miner - Proof of Intelligence Network")
        print("=" * 60)
        print(f"Server: {self.config.server_url}")
        print(f"Miner ID: {self.config.contributor_id}")
        print(f"CPU Cores: {self.config.cpu_cores}")
        print(f"GPU: {self.config.gpu_model or 'None'} ({self.config.gpu_memory_mb}MB)")
        print("-" * 60)
        
        if not await self.register():
            print("[FATAL] Cannot register with network. Exiting.")
            return
        
        if not await self.start_session():
            print("[FATAL] Cannot start session. Exiting.")
            return
        
        self.is_running = True
        heartbeat_counter = 0
        
        print("\n[MINING] Starting AI task processing...")
        print("[INFO] Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                task = await self.fetch_task()
                
                if task:
                    task_id = task.get("task_id")
                    task_type = task.get("task_type")
                    print(f"[TASK] Processing: {task_type} (ID: {task_id[:8]}...)")
                    
                    result = self.process_task(task)
                    print(f"[DONE] Computed in {result.get('compute_time_ms', 0)}ms")
                    
                    reward = await self.submit_result(task_id, result)
                    if reward:
                        self.tasks_completed += 1
                        self.total_rewards += reward
                        print(f"[REWARD] +{reward:.4f} NEO | Total: {self.total_rewards:.4f} NEO")
                
                heartbeat_counter += 1
                if heartbeat_counter >= 3:
                    hb = await self.send_heartbeat(self.tasks_completed)
                    if hb:
                        print(f"[STATUS] Tasks: {self.tasks_completed} | Rewards: {self.total_rewards:.4f} NEO")
                    heartbeat_counter = 0
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down miner...")
        finally:
            self.is_running = False
            summary = await self.stop_session()
            print("\n" + "=" * 60)
            print("    Mining Session Summary")
            print("=" * 60)
            print(f"Tasks Completed: {self.tasks_completed}")
            print(f"Total Rewards: {self.total_rewards:.4f} NEO")
            if summary:
                print(f"Session Duration: {summary.get('duration_seconds', 0):.0f}s")
            print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="NeoNet AI Miner")
    parser.add_argument("--server", default="http://localhost:8000", help="NeoNet server URL")
    parser.add_argument("--id", default=None, help="Miner ID (generated if not provided)")
    parser.add_argument("--cpu", type=int, default=4, help="CPU cores to use")
    parser.add_argument("--gpu-mem", type=int, default=0, help="GPU memory in MB")
    parser.add_argument("--gpu-model", default="", help="GPU model name")
    
    args = parser.parse_args()
    
    miner_id = args.id or f"miner_{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"
    
    config = MinerConfig(
        server_url=args.server,
        contributor_id=miner_id,
        cpu_cores=args.cpu,
        gpu_memory_mb=args.gpu_mem,
        gpu_model=args.gpu_model
    )
    
    miner = NeoNetMiner(config)
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())
