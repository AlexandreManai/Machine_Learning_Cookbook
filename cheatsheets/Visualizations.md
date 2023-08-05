# Visualizations Cheat Sheet

## Data correlation 

```python
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, ax=ax) # data needs to be numeric
```

## Data distributions

### Boxplots

```python
fig, axs = plt.subplots(ncols=7, nrows=1, figsize=(10,5))
index = 0
axs = axs.flatten()
for k, v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=.4, w_pad=.5, h_pad=1)
```

### Histograms
```python
fig, axs = plt.subplots(ncols=7, nrows=1, figsize=(10,5))
index = 0
axs = axs.flatten()
for k, v in numeric_data.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=.4, w_pad=.5, h_pad=1)
```
