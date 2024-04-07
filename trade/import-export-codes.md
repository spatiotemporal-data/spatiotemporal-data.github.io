---
layout: default
---

# Utilizing International Import and Export Trade Data from WTO Stats

- For reproducing Figure ...

<br>

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 2.5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.sum(mat, axis = 0), color = 'red', linewidth = 1.8) # Imports
plt.plot(np.sum(mat1, axis = 0), color = 'blue', linewidth = 1.8) # Exports
plt.xticks(np.arange(0, 23, 1))
plt.xlabel('Year')
plt.ylabel('Trade values (Million US dollar)')
plt.grid(axis = 'both', linestyle='dashed', linewidth = 0.1, color = 'gray')
ax.tick_params(direction = "in")
ax.set_xlim([-0.5, 22.5])
ticks_data = np.arange(2000, 2023, 1)
plt.xticks(np.arange(23), ticks_data, rotation = 90)
plt.legend(['Imports', 'Exports'])
plt.show()
fig.savefig("import_export_world.png", bbox_inches = "tight")
```

<br>

- For reproducing Figure ...

<br>

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 2.5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(mat[203, :], color = 'red', linewidth = 1.8) # USA Imports
plt.plot(mat1[203, :], color = 'blue', linewidth = 1.8) # USA Exports
plt.xticks(np.arange(0, 23, 1))
plt.xlabel('Year')
plt.ylabel('Trade values (Million US dollar)')
plt.grid(axis = 'both', linestyle='dashed', linewidth = 0.1, color = 'gray')
ax.tick_params(direction = "in")
ax.set_xlim([-0.5, 22.5])
ticks_data = np.arange(2000, 2023, 1)
plt.xticks(np.arange(23), ticks_data, rotation = 90)
plt.legend(['Imports', 'Exports'])
plt.show()
fig.savefig("import_export_usa.png", bbox_inches = "tight")
```

<br>

- For reproducing Figure ...

<br>

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 2.5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(mat[203, :] / np.sum(mat, axis = 0),
         color = 'red', linewidth = 1.8) # USA Imports
plt.plot(mat1[203, :] / np.sum(mat1, axis = 0),
         color = 'blue', linewidth = 1.8) # USA Exports
plt.xticks(np.arange(0, 23, 1))
plt.xlabel('Year')
plt.ylabel('USA trade percentages')
plt.grid(axis = 'both', linestyle='dashed', linewidth = 0.1, color = 'gray')
ax.tick_params(direction = "in")
ax.set_xlim([-0.5, 22.5])
ticks_data = np.arange(2000, 2023, 1)
plt.xticks(np.arange(23), ticks_data, rotation = 90)
plt.legend(['Imports', 'Exports'])
plt.show()
fig.savefig("import_export_percentage_usa.png", bbox_inches = "tight")
```

<br>

- For reproducing Figure ...

<br>

```python

```

<br>

- For reproducing Figure ...

<br>

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 2.5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(mat[38, :] / np.sum(mat, axis = 0),
         color = 'red', linewidth = 1.8) # China Imports
plt.plot(mat1[38, :] / np.sum(mat1, axis = 0),
         color = 'blue', linewidth = 1.8) # China Exports
plt.xticks(np.arange(0, 23, 1))
plt.xlabel('Year')
plt.ylabel('China trade percentages')
plt.grid(axis = 'both', linestyle='dashed', linewidth = 0.1, color = 'gray')
ax.tick_params(direction = "in")
ax.set_xlim([-0.5, 22.5])
ticks_data = np.arange(2000, 2023, 1)
plt.xticks(np.arange(23), ticks_data, rotation = 90)
plt.legend(['Imports', 'Exports'])
plt.show()
fig.savefig("import_export_percentage_chn.png", bbox_inches = "tight")
```

<br>


<br>
<p align="left">(Posted by <a href="https://xinychen.github.io/">Xinyu Chen</a> on April 6, 2024.)</p>
