import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def visualize_scatter_with_images(X_2d_data, images, figsize=(32,24), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        i= str(i)
        i='/data/dataD/snapshot'+i[38:-1]
        print(i)
        try:
          i = Image.open(i)
          i.thumbnail((64, 48), Image.ANTIALIAS)
          img = OffsetImage(i, zoom=image_zoom)
          ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
          artists.append(ax.add_artist(ab))
        except: pass
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()



X= np.asarray(np.load('all_feats.npy'))[0:4000]
P= np.load('all_urls.npy')[0:4000]
P= np.hstack(P)
Y= TSNE(n_components=2).fit_transform(X)
visualize_scatter_with_images(Y,P)
#X = StandardScaler().fit_transform(X)

#Y = cdist(X, X, 'euclidean')
#n, bins, patches = plt.hist(X, 50, normed=1, facecolor='green', alpha=0.75)
#print(n,bins)
#l = plt.plot(bins, y, 'r--', linewidth=1)
#plt.show()
#db = DBSCAN(eps=150, min_samples=10).fit(X)
#print(db.labels_)
#km= KMeans(n_clusters=10).fit(X)
#print(km.labels_)
