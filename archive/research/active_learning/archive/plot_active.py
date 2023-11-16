import numpy as np
import matplotlib.pyplot as plt

names = ['Confidence','Entropy','Informative Diverse','K-center','Margin','Margin Cluster Mean','Random']
colors= ['blue','red','green','yellow','red','purple','black']
data = np.loadtxt("all_processed.data",delimiter=",",skiprows=1)
x = np.asarray(range(10,301))*100
plt.xlim(1000,30000)
plt.xticks(np.arange(1000, 31000, 1000), [str(int(x/1000))+'K' for x in np.arange(1000, 31000, 1000)], rotation=90)
plt.yticks(np.arange(0.78, 0.95, 0.01))
plt.xlabel("Number of Queries",fontweight='bold',fontsize=18)
plt.ylabel("Accuracy",fontweight='bold',fontsize=18)
for i in range(len(names)):
    if i!=1:
        plt.plot(x, data[:,i], color = colors[i], linewidth = 2, label = names[i])
plt.hlines([0.909], 0, 30000, colors=['orange'], linestyles=['dashed'], linewidth=3)
plt.text(4000, 0.909 + 0.002, 'Norouzzadeh et al. 2018', fontweight='bold', fontsize=12)
plt.grid(True)
plt.legend(loc=4)
plt.show()
