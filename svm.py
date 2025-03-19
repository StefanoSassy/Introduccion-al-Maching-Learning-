import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_circles
from sklearn import svm

# Generar datos
X, c = make_circles(n_samples=500, noise=0.09)
z = X[:, 0]**2 + X[:, 1]**2

# Entrenar SVM
features = np.concatenate((X, z.reshape(-1, 1)), axis=1)
clf = svm.SVC(kernel='linear')
clf.fit(features, c)

# Definir el plano separado por SVM
x3 = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

# Crear malla de puntos para el plano
tmp = np.linspace(-1.5, 1.5, 100)
x, y = np.meshgrid(tmp, tmp)
z_plane = x3(x, y)

# Crear figura interactiva con Plotly
fig = go.Figure()

# Agregar puntos de la clase 0 (rojos)
fig.add_trace(go.Scatter3d(x=X[c == 0, 0], y=X[c == 0, 1], z=z[c == 0],
                           mode='markers', marker=dict(size=5, color='red'), name='Clase 0'))

# Agregar puntos de la clase 1 (verdes)
fig.add_trace(go.Scatter3d(x=X[c == 1, 0], y=X[c == 1, 1], z=z[c == 1],
                           mode='markers', marker=dict(size=5, color='green'), name='Clase 1'))

# Agregar el plano de separación
fig.add_trace(go.Surface(x=x, y=y, z=z_plane, colorscale='Blues', opacity=0.5, name='Plano SVM'))

# Configurar la vista
fig.update_layout(scene=dict(xaxis_title="X1", yaxis_title="X2", zaxis_title="Z"),
                  title="Separación con SVM en 3D")

fig.show()