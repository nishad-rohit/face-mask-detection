import os

packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "opencv-python",
    "scikit-learn",
    "tensorflow"
]

for pkg in packages:
    os.system(f"pip install {pkg}")
