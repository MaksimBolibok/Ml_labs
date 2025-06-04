
import numpy as np
import pandas as pd

n_samples = 100
noise = 0.15

def generate_circle(radius, label):
    angles = 2 * np.pi * np.random.rand(n_samples)
    x = np.cos(angles) * radius + np.random.normal(0, noise, n_samples)
    y = np.sin(angles) * radius + np.random.normal(0, noise, n_samples)
    labels = np.full(n_samples, label)
    return np.stack([labels, x, y], axis=1)

def generate_background(label):
    x = np.random.uniform(-4, 4, n_samples * 2)
    y = np.random.uniform(-4, 4, n_samples * 2)
    mask = (x**2 + y**2) > (2.8)**2
    x, y = x[mask][:n_samples], y[mask][:n_samples]
    labels = np.full(n_samples, label)
    return np.stack([labels, x, y], axis=1)

circle1 = generate_circle(1.0, 0)
circle2 = generate_circle(2.5, 1)
background = generate_background(2)

data = np.concatenate([circle1, circle2, background])
df = pd.DataFrame(data, columns=["Label", "x", "y"])
df = df.sample(frac=1).reset_index(drop=True)

df[:210].to_csv("train_multiclass.csv", index=False, header=False)
df[210:].to_csv("test_multiclass.csv", index=False, header=False)

print("CSV файли збережено.")
