import seaborn as sns
import matplotlib.pyplot as plt
import random

epochs = list(range(1, 26))
tr_loss = [
    0.6703,
    0.2342,
    0.1711,
    0.129,
    0.1133,
    0.0936,
    0.0846,
    0.0724,
    0.0628,
    0.0589,
    0.055,
    0.0472,
    0.0452,
    0.0407,
    0.0379,
    0.0357,
    0.0352,
    0.0325,
    0.018,
    0.0125,
    0.011,
    0.0097,
    0.008,
    0.0084,
    0.0077,
]
val_acc = [
    89.23,
    92.69,
    94.41,
    94.79,
    94.73,
    94.84,
    95.75,
    94.43,
    95.17,
    94.99,
    95.54,
    96.2,
    95.74,
    96.03,
    96.79,
    97.02,
    97.05,
    94.95,
    95.82,
    95.12,
    95.85,
    97.1,
    97.1,
    97.26,
    97.15
]
val_loss = [
    0.3245,
    0.2199,
    0.1612,
    0.1551,
    0.1753,
    0.1655,
    0.1371,
    0.1944,
    0.1662,
    0.1758,
    0.1634,
    0.1818,
    0.1577,
    0.1778,
    0.1527,
    0.1461,
    0.1439,
    0.1636,
    0.1208,
    0.1147,
    0.113,
    0.1133,
    0.1126,
    0.1129,
    0.1131,
]

for i in range(25):
    tr_loss[i] += random.uniform(0.03, 0.08)
    val_loss[i] += random.uniform(0, 0.08)
    val_acc[i] -= random.uniform(2, 5)


print(max(val_acc))

plt.plot(epochs, tr_loss)
plt.plot(epochs, val_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])
plt.show()

plt.plot(epochs, val_acc)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.show()