# Visualization Reference (matplotlib + seaborn)

## matplotlib Object-Oriented API (Recommended)

Always use the OO interface for production code:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
x = np.linspace(0, 2*np.pi, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Trigonometric Functions')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

## Plot Types

### Line plots - Time series, continuous data
```python
ax.plot(x, y, linewidth=2, linestyle='--', marker='o', color='blue')
```

### Scatter plots - Relationships, correlations
```python
ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, cmap='viridis')
```

### Bar charts - Categorical comparisons
```python
ax.bar(categories, values, color='steelblue', edgecolor='black')
ax.barh(categories, values)  # horizontal
```

### Histograms - Distributions
```python
ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
```

### Heatmaps - Matrices, correlations
```python
im = ax.imshow(matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im, ax=ax)
```

### Box plots - Statistical distributions
```python
ax.boxplot([data1, data2, data3], labels=['A', 'B', 'C'])
```

### Violin plots - Distribution densities
```python
ax.violinplot([data1, data2, data3], positions=[1, 2, 3])
```

### Radar/Spider charts
```python
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
values += values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, alpha=0.25)
ax.plot(angles, values, linewidth=2)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
```

## Multiple Subplots

### Regular grid
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y1)
axes[0, 1].scatter(x, y2)
```

### Mosaic layout (flexible)
```python
fig, axes = plt.subplot_mosaic([['left', 'right_top'],
                                 ['left', 'right_bottom']],
                                figsize=(10, 8))
```

### GridSpec (maximum control)
```python
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])      # top row, all columns
ax2 = fig.add_subplot(gs[1:, 0])     # bottom rows, first column
ax3 = fig.add_subplot(gs[1:, 1:])    # bottom rows, last columns
```

## seaborn Integration

```python
import seaborn as sns

# Configure for publication
sns.set_theme(style='ticks', context='paper', font_scale=1.1)
sns.set_palette('colorblind')

# Statistical plots with automatic CIs
fig, ax = plt.subplots(figsize=(3.5, 3))
sns.boxplot(data=df, x='treatment', y='response', palette='Set2', ax=ax)
sns.stripplot(data=df, x='treatment', y='response',
              color='black', alpha=0.3, size=3, ax=ax)
sns.despine()

# Line plot with confidence bands
sns.lineplot(data=df, x='time', y='measurement',
             hue='treatment', errorbar=('ci', 95), markers=True)

# Correlation heatmap
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True)

# Faceted plots
g = sns.relplot(data=df, x='dose', y='response',
                hue='treatment', col='cell_line',
                kind='line', height=2.5, aspect=1.2)
```

## Color Palettes

### Colorblind-safe (always preferred)
```python
# Okabe-Ito palette
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=okabe_ito)

# Or use seaborn
sns.set_palette('colorblind')
```

### Colormap selection
- **Sequential** (viridis, plasma, inferno): ordered data
- **Diverging** (coolwarm, RdBu): data with meaningful center
- **Qualitative** (tab10, Set3): categorical data
- **NEVER use jet or rainbow**: not perceptually uniform

## Saving Figures

```python
# High-res PNG for publications
plt.savefig('figure.png', dpi=300, bbox_inches='tight', facecolor='white')

# Vector for journals
plt.savefig('figure.pdf', bbox_inches='tight')
plt.savefig('figure.svg', bbox_inches='tight')

# Transparent background
plt.savefig('figure.png', dpi=300, bbox_inches='tight', transparent=True)
```

### DPI guidelines
- Screen/notebook: 72-100 dpi
- Web: 150 dpi
- Print/publications: 300 dpi
- Line art: 600+ dpi

## Publication Figure Checklist

- [ ] Figure size matches journal specs (Nature single: 89mm, double: 183mm)
- [ ] Font >= 6pt at final print size, sans-serif (Arial, Helvetica)
- [ ] Colorblind-friendly palette
- [ ] Works in grayscale
- [ ] All axes labeled with units
- [ ] Error bars present with type noted in caption
- [ ] Panel labels (A, B, C) bold, consistent
- [ ] 300+ DPI, vector format preferred
- [ ] No chart junk (unnecessary grids, 3D effects)

## 3D Plots

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
```

## Common Gotchas

1. **Overlapping elements**: Use `constrained_layout=True`
2. **Memory with many figures**: Close with `plt.close(fig)`
3. **DPI confusion**: figsize is inches, pixels = dpi * inches
4. **State confusion**: Use OO interface to avoid pyplot issues
