import time
import torch

# Set TF32 precision settings
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

sizes = [4096, 8192]
dtypes = [torch.float32, torch.float16, torch.bfloat16]
NUM_ITERATIONS = 10 

print(f"\n🔥 RTX 5080 (sm_120) Throughput Test 🔥\n{'='*60}")

for n in sizes:
    print(f"\nMatrix size: {n}x{n}")
    for dtype in dtypes:
        dtype_str = str(dtype).split('.')[-1].upper()

        x = torch.randn(n, n, device='cuda', dtype=dtype)
        y = torch.randn(n, n, device='cuda', dtype=dtype)

        # Warmup
        z = x @ y 

        torch.cuda.synchronize()
        t0 = time.time()

        # Benchmark loop
        for _ in range(NUM_ITERATIONS):
            z = x @ y

        torch.cuda.synchronize()

        tflops = NUM_ITERATIONS * 2 * n**3 / (time.time() - t0) / 1e12

        print(f"  {dtype_str:<8} → {tflops:8.2f} TFLOPS")
print(f"\n{'='*60}\nBenchmark completed.\n")