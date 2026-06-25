import matplotlib.pyplot as plt
import numpy as np

data = np.load("mobifall_data.npz")

print("Keys:", data.files)
print("acc:", data['acc'].shape)
print("gyro:", data['gyro'].shape)
print("labels:", data['labels'].shape)

print("Unique subjects:", len(np.unique(data['subject_ids'])))
print("Fall:", (data['labels'] == 1).sum())
print("ADL:", (data['labels'] == 0).sum())


idx = np.where(data['labels'] == 0)[0][0]
sample = data['acc'][idx]

plt.plot(sample[:, 0], label='x')
plt.plot(sample[:, 1], label='y')
plt.plot(sample[:, 2], label='z')
plt.legend()
plt.title("Accelerometer sample")
plt.show()

idx = np.where(data['labels'] == 1)[0][0]
sample = data['acc'][idx]

plt.plot(sample[:, 0], label='x')
plt.plot(sample[:, 1], label='y')
plt.plot(sample[:, 2], label='z')
plt.legend()
plt.title("Accelerometer sample")
plt.show()