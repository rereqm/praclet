import matplotlib.pyplot as plt
from src.evaluation.benchmark_data import RESULTS

models = list(RESULTS.keys())
fps = [RESULTS[m]["fps"] for m in models]
map50 = [RESULTS[m]["map50"] for m in models]

plt.figure()
plt.bar(models, fps)
plt.title("FPS comparison")
plt.show()

plt.figure()
plt.bar(models, map50)
plt.title("mAP50 comparison")
plt.show()