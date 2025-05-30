# import lightgbm as lgb
# print(lgb.basic._InnerPredictor.__doc__)  # Juste pour forcer l'import de base
# print(lgb.__path__)  # pour confirmer le bon chemin

# import subprocess
# print(subprocess.run(["lightgbm", "--gpu-platform-id=0", "--gpu-device-id=0"], capture_output=True, text=True).stderr)


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)
print("LightGBM path:", lgb.__file__)
print("GPU enabled:", hasattr(lgb.basic, "_InnerPredictor") and "gpu" in lgb.__file__)
