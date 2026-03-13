import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../report/tables/ablation_results.csv")

plt.figure()
plt.plot(df["value"], df["val_accuracy"], marker="o")
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.title("DistilBERT Learning Rate Ablation")

plt.savefig("../../report/figures/ablation_learning_rate.png")
plt.close()