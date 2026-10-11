import torch
from ultralytics import SAM


print("=" * 70)
print("SAM 2.1 Environment Test")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("\nLoading SAM 2.1 Base...")

model = SAM("sam2.1_b.pt")

print("\nSAM 2.1 Base loaded successfully.")

print("\nModel information:")
model.info()

print("\nEnvironment test PASSED.")