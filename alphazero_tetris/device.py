import jax

# First, let's get explicit device lists for both platforms
try:
    gpu_devices = jax.devices("gpu")  # Directly request GPU devices
except:
    gpu_devices = []
cpu_devices = jax.devices("cpu")  # Directly request CPU devices

# Now we can check GPU availability based on the actual list length
has_gpu = len(gpu_devices) > 0

# Assign devices with explicit error handling
training_device = gpu_devices[0] if has_gpu else cpu_devices[0]
collect_device = cpu_devices[0]  # Always use CPU for collection
