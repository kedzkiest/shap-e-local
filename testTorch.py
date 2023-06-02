import torch

# Check if PyTorch is using the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using: {device}")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Test a simple PyTorch tensor operation on the GPU
x = torch.tensor([2.0, 3.0], device=device)
y = torch.tensor([4.0, 5.0], device=device)
result = x * y
print(f"Result of tensor operation: {result}")