from pathlib import Path
import joblib

PROJECT_ROOT = Path(r"D:\AI-Personalized-Learning-Recommendation")
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_FILE = MODEL_DIR / "hybrid_config.joblib"

CONFIG = {
    "skill_weight": 0.40,
    "content_weight": 0.30,
    "collaborative_weight": 0.15,
    "performance_weight": 0.15,
}

print("=" * 70)
print("REBUILDING HYBRID CONFIGURATION")
print("=" * 70)

print(f"\nOutput file:")
print(CONFIG_FILE)

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Remove the corrupted/empty file first.
if CONFIG_FILE.exists():
    print("\nRemoving existing hybrid_config.joblib...")
    CONFIG_FILE.unlink()

# Save the correct configuration.
joblib.dump(
    CONFIG,
    CONFIG_FILE
)

print("\nConfiguration saved.")

# Verify immediately.
if not CONFIG_FILE.exists():
    raise RuntimeError("File was not created.")

file_size = CONFIG_FILE.stat().st_size

print(f"File size: {file_size} bytes")

if file_size <= 0:
    raise RuntimeError("Configuration file is empty.")

# Load it back.
loaded_config = joblib.load(CONFIG_FILE)

print("\nLoaded configuration:")
print(loaded_config)

if loaded_config != CONFIG:
    raise RuntimeError(
        "Saved configuration does not match expected configuration."
    )

print("\n" + "=" * 70)
print("HYBRID CONFIGURATION REBUILT SUCCESSFULLY")
print("=" * 70)