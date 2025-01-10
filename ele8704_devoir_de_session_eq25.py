#!/usr/bin/env python
# coding: utf-8

# # Projet ELE8704
# 

# ## Importation des librairies et mise en place du projet

# In[125]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from scapy.all import rdpcap,UDP, TCP, IP
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import ipaddress
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve







pd.options.mode.chained_assignment = None  


# On enregistre nos adresses IP pour wifi et éthernet

# In[126]:


my_ip_wifi="192.168.2.35"
my_ip_ethernet="192.168.2.68"


# In[127]:


def TCP_control(packet):
    if TCP in packet and hasattr(packet[TCP], 'flags'):
        # Check if any control flags are set (e.g., SYN, ACK, RST, FIN)
        control_flags = packet[TCP].flags & (0x02 | 0x10 | 0x04 | 0x01)
        return control_flags != 0 and control_flags != 0x10  # Exclude ACK-only packets
    return False


# ## Vidéo Wifi avec Youtube

# In[128]:


pcap_file = r'VideoWifi.pcapng'
data=rdpcap(pcap_file)
print('#################################################################')
print('On commence avec la vidéo')
# On filtre pour ne garder que les paquets IP dans notre analyse. On va aussi dès maintenant enregistrer les protocoles et décider si il s'agit de paquets de contrôle ou non.

# In[1]:


protocol_names = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
}


# In[2]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
    else:
        control_flag = False

    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        proto_couche_sup=protocol_names.get(p.proto)
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })


# On transforme maintenant nos données en dataframe pour pouvoir les analyser

# In[131]:


df_video_wifi=pd.DataFrame(p_data)
df_video_wifi.sort_values(by='Time')
df_video_wifi['Protocol Transport'].value_counts()


# ### Temps d'Arrivée

# On nous demande alors d'analyser les temps d'arrivée des paquets de données. On crée un dataframe qui ne contient que les paquets de données. 

# In[132]:


df_video_wifi_no_control = df_video_wifi[df_video_wifi.index.isin(df_video_wifi[df_video_wifi['Control Flag'] == False].index)]


# On représente l'histogramme des temps d'arrivée

# In[133]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_wifi_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("VideoWifiArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[134]:


# Calculer les statistiques de base
data=df_video_wifi_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques de Vidéo par Wifi')
print('Statistiques pour le temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('===========================================================')


# ### Longueur paquets

# In[135]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_video_wifi['Protocol'].unique()
data_by_protocol = [df_video_wifi[df_video_wifi['Protocol'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("taillePaquetsWifi.pdf", format="pdf")
plt.show()


# In[166]:



# Exemple de données (remplacez par vos propres données)
data = df_video_wifi['Length']

# Ajuster un modèle de mélange de Gaussiennes (2 composantes)
gmm = GaussianMixture(n_components=2)
gmm.fit(data.to_numpy().reshape(-1, 1))

# Générer des échantillons de la distribution ajustée
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
pdf_gmm = np.exp(gmm.score_samples(x))  # Probabilité d'appartenance à chaque composant

# Effectuer un test KS sur les données et la distribution ajustée
ks_statistic, p_value = stats.ks_2samp(data, x.flatten())

# Afficher la p-valeur
print(f"p-valeur du test KS : {p_value}")

# Tracer l'histogramme des données et la courbe de la distribution ajustée
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogramme des données')
plt.plot(x, pdf_gmm, label='Ajustement - Mélange de 2 Gaussiennes', color='r', linewidth=2)

# Ajouter les titres et la légende
plt.title('Histogramme et ajustement du modèle bimodal', fontsize=14)
plt.xlabel('Valeurs', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.legend()
plt.savefig('VideoTailleWifiAjustée.pdf', format='pdf')

# Afficher le graphique
plt.show()


# In[137]:


# Calculer les statistiques de base
data=df_video_wifi['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour la longueur des paquets')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('================================================================')

# ### Analyse de la Gigue

# On ne considère ici que les paquets de données car la gigue concerne les temps d'inter-arrivée.

# In[161]:


df_video_wifi_no_control.sort_values(by='Time')
df_video_wifi_no_control['Inter-Arrival Time']=df_video_wifi_no_control['Time'].diff().fillna(0)
df_video_wifi_no_control['Inter-Arrival Time'] = pd.to_numeric(df_video_wifi_no_control['Inter-Arrival Time'], errors='coerce')
df_video_wifi_no_control['Inter-Arrival Time'] = df_video_wifi_no_control['Inter-Arrival Time'].clip(lower=0)


# In[168]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_wifi_no_control['Inter-Arrival Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Inter-Arrival Time")

plt.title("Histogram of Inter-Arrival Time", fontsize=14)
plt.xlabel("Inter-Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("VideoWifiIATUntreated.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[169]:



# Données : temps d'inter-arrivée
data = df_video_wifi_no_control['Inter-Arrival Time'].values
print(data)
# Créer l'histogramme des données
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data), np.std(data)
norm_pdf = stats.norm.pdf(np.linspace(data.min(), data.max(), 1000), mu, sigma)
plt.plot(np.linspace(data.min(), data.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data)
exp_pdf = stats.expon.pdf(np.linspace(data.min(), data.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data.min(), data.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("VideoWifiFits.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()



# In[141]:


# Calculer les statistiques de base
data=df_video_wifi_no_control['Inter-Arrival Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
zero_percentage = (data == 0).mean() * 100  # Pourcentage de zéro
# Afficher les résultats
print('Statistiques des temps d inter-arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print(f"Pourcentage de valeurs égales à zéro: {zero_percentage:.2f}%")
print('==========================================================')

# In[170]:


df_video_wifi_no_control_no0 = df_video_wifi_no_control[df_video_wifi_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_video_wifi_no_control_no0[df_video_wifi_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("VideoWifiFitFiltered.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()




# ## Vidéo Youtube avec Ethernet

# In[143]:


pcap_file = r'VideoEthernet.pcapng'
data=rdpcap(pcap_file)


# In[144]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]

# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
    else:
        control_flag = False

    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol': proto_name,  # Utilisation du nom ici
            'Length': len(p),
            'Control Flag': control_flag,
        })


# In[145]:


df_video_ethernet=pd.DataFrame(p_data)
df_video_ethernet


# ### Arrivées

# In[146]:


df_video_ethernet_no_control = df_video_ethernet[df_video_ethernet.index.isin(df_video_ethernet[df_video_ethernet['Control Flag'] == False].index)]


# In[147]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_ethernet_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("VideoEthernetArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[148]:


# Calculer les statistiques de base
data=df_video_ethernet_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour la vidéo par Ethernet')
print('Statistiques pour les temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('===========================================================')

# ### Longueur des paquets

# In[149]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_video_ethernet['Protocol'].unique()
data_by_protocol = [df_video_ethernet[df_video_ethernet['Protocol'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("taillePaquetsEthernet.pdf", format="pdf")
plt.show()


# In[167]:


# Exemple de données (remplacez par vos propres données)
data = df_video_ethernet['Length']

# Ajuster un modèle de mélange de Gaussiennes (2 composantes)
gmm = GaussianMixture(n_components=2)
gmm.fit(data.to_numpy().reshape(-1, 1))

# Générer des échantillons de la distribution ajustée
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
pdf_gmm = np.exp(gmm.score_samples(x))  # Probabilité d'appartenance à chaque composant

# Effectuer un test KS sur les données et la distribution ajustée
ks_statistic, p_value = stats.ks_2samp(data, x.flatten())

# Afficher la p-valeur
print(f"p-valeur du test KS : {p_value}")

# Tracer l'histogramme des données et la courbe de la distribution ajustée
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogramme des données')
plt.plot(x, pdf_gmm, label='Ajustement - Mélange de 2 Gaussiennes', color='r', linewidth=2)

# Ajouter les titres et la légende
plt.title('Histogramme et ajustement du modèle bimodal', fontsize=14)
plt.xlabel('Valeurs', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.legend()
plt.savefig('VideoTailleEthernetAjustée.pdf', format='pdf')

# Afficher le graphique
plt.show()


# In[150]:


# Calculer les statistiques de base
data=df_video_ethernet['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques de longueur des paquets')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('=================================================================================')

# ### Gigue

# In[151]:


df_video_ethernet_no_control['Inter-Arrival Time']=df_video_ethernet_no_control['Time'].diff().fillna(0)
df_video_ethernet_no_control['Inter-Arrival Time'] = pd.to_numeric(df_video_ethernet_no_control['Inter-Arrival Time'], errors='coerce')
df_video_ethernet_no_control['Inter-Arrival Time'] = df_video_ethernet_no_control['Inter-Arrival Time'].clip(lower=0)


# In[174]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_ethernet_no_control['Inter-Arrival Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Inter-Arrival Time")

plt.title("Histogram of Inter-Arrival Time", fontsize=14)
plt.xlabel("Inter-Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("VideoEthernetIAT.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[175]:
# Données : temps d'inter-arrivée
data = df_video_ethernet_no_control['Inter-Arrival Time'].values

# Créer l'histogramme des données
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data), np.std(data)
norm_pdf = stats.norm.pdf(np.linspace(data.min(), data.max(), 1000), mu, sigma)
plt.plot(np.linspace(data.min(), data.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data)
exp_pdf = stats.expon.pdf(np.linspace(data.min(), data.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data.min(), data.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("VideoEthernetFit.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()



# In[154]:


# Calculer les statistiques de base
data=df_video_ethernet_no_control['Inter-Arrival Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
zero_percentage = (data == 0).mean() * 100  # Pourcentage de zéro
# Afficher les résultats
print('Statistiques des temps d inter-arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print(f"Pourcentage de valeurs égales à zéro: {zero_percentage:.2f}%")
print('====================================================================')

# In[176]:


df_video_ethernet_no_control_no0 = df_video_ethernet_no_control[df_video_ethernet_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_video_ethernet_no_control_no0[df_video_ethernet_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("VideoEthernetFitFiltered.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()

#############################################################################################################################################################################
# ## Téléchargement en Wifi

# In[61]:


pcap_file = r'DownloadWifi.pcapng'
data=rdpcap(pcap_file)
print('#################################################################')
print('On enchaine avec le téléchargement')
# On filtre pour ne garder que les paquets IP dans notre analyse. On va aussi dès maintenant enregistrer les protocoles et décider si il s'agit de paquets de contrôle ou non.

# In[62]:


protocol_names = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    
    # Ajoute d'autres protocoles si nécessaire
}


# In[63]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
index=0
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
    else:
        control_flag = False

    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        list.append(index)
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        proto_couche_sup=protocol_names.get(p.proto)
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    index+=1


# On transforme maintenant nos données en dataframe pour pouvoir les analyser

# In[64]:


df_download_wifi=pd.DataFrame(p_data)
df_download_wifi['Protocol Transport'].value_counts()


# ### Temps d'Arrivée

# On nous demande alors d'analyser les temps d'arrivée des paquets de données. On crée un dataframe qui ne contient que les paquets de données. 

# In[65]:


df_download_wifi_no_control = df_download_wifi[df_download_wifi.index.isin(df_download_wifi[df_download_wifi['Control Flag'] == False].index)]


# On représente l'histogramme des temps d'arrivée

# In[66]:


plt.figure(figsize=(12, 6))
plt.hist(df_download_wifi_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("DownloadWifiArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[67]:


# Calculer les statistiques de base
data=df_download_wifi_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour le télechargement en WIFI')
print('Statistiques des temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('================================================================================')


# ### Longueur paquets

# In[68]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_download_wifi['Protocol'].unique()
data_by_protocol = [df_download_wifi[df_download_wifi['Protocol'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("Download_taillePaquetsWifi.pdf", format="pdf")
plt.show()


# In[69]:




# Exemple de données (remplacez par vos propres données)
data = df_download_wifi['Length']

# Ajuster un modèle de mélange de Gaussiennes (2 composantes)
gmm = GaussianMixture(n_components=2)
gmm.fit(data.to_numpy().reshape(-1, 1))

# Générer des échantillons de la distribution ajustée
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
pdf_gmm = np.exp(gmm.score_samples(x))  # Probabilité d'appartenance à chaque composant

# Effectuer un test KS sur les données et la distribution ajustée
ks_statistic, p_value = stats.ks_2samp(data, x.flatten())

# Afficher la p-valeur
print(f"p-valeur du test KS : {p_value}")

# Tracer l'histogramme des données et la courbe de la distribution ajustée
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogramme des données')
plt.plot(x, pdf_gmm, label='Ajustement - Mélange de 2 Gaussiennes', color='r', linewidth=2)

# Ajouter les titres et la légende
plt.title('Histogramme et ajustement du modèle bimodal', fontsize=14)
plt.xlabel('Valeurs', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.legend()

# Afficher le graphique
plt.show()


# In[70]:


# Calculer les statistiques de base
data=df_download_wifi['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
print('Statistiques de longueur des paquets')
# Afficher les résultats
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('================================================================================')


# ### Analyse du temps d'inter-arrivée

# On ne considère ici que les paquets de données.

# In[71]:


df_download_wifi_no_control['Inter-Arrival Time']=df_download_wifi_no_control['Time'].diff().fillna(0)
df_download_wifi_no_control['Inter-Arrival Time'] = pd.to_numeric(df_download_wifi_no_control['Inter-Arrival Time'], errors='coerce')


# In[72]:


df_download_wifi_no_control_no0 = df_download_wifi_no_control[df_download_wifi_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_download_wifi_no_control_no0[df_download_wifi_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("download_normal_and_exponential_fits_filtered.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()




# ## Vidéo Youtube avec Ethernet

# In[73]:


pcap_file = r'DownloadEthernet.pcapng'
data=rdpcap(pcap_file)


# In[74]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]

# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
    else:
        control_flag = False

    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol': proto_name,  # Utilisation du nom ici
            'Length': len(p),
            'Control Flag': control_flag,
        })


# In[75]:


df_video_ethernet=pd.DataFrame(p_data)
df_video_ethernet


# ### Arrivées

# In[76]:


df_video_ethernet_no_control = df_video_ethernet[df_video_ethernet.index.isin(df_video_ethernet[df_video_ethernet['Control Flag'] == False].index)]


# In[77]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_ethernet_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("DownloadEthernetArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[78]:


# Calculer les statistiques de base
data=df_video_ethernet_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour le téléchargement par Ethernet')
print('Statistiques pour les temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('================================================================================')

# ### Longueur des paquets

# In[79]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_video_ethernet['Protocol'].unique()
data_by_protocol = [df_video_ethernet[df_video_ethernet['Protocol'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("DownloadtaillePaquetsEthernet.pdf", format="pdf")
plt.show()


# In[82]:


# Calculer les statistiques de base
data=df_video_ethernet['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les 
print('Statistiques de longueur des paquets')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('================================================================================')

# In[80]:


df_video_ethernet_no_control['Inter-Arrival Time']=df_video_ethernet_no_control['Time'].diff().fillna(0)
df_video_ethernet_no_control['Inter-Arrival Time'] = pd.to_numeric(df_video_ethernet_no_control['Inter-Arrival Time'], errors='coerce')


# In[81]:


df_video_ethernet_no_control_no0 = df_video_ethernet_no_control[df_video_ethernet_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_video_ethernet_no_control_no0[df_video_ethernet_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("Download_ethernet_normal_and_exponential_fits_filtered.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()
################################################################################################################################

pcap_file = r'ConvWifi.pcapng'
data=rdpcap(pcap_file)
print('#################################################################')
print('On termine avec la conversation')

# On filtre pour ne garder que les paquets IP dans notre analyse. On va aussi dès maintenant enregistrer les protocoles et décider si il s'agit de paquets de contrôle ou non.

# In[230]:


protocol_names = {
    17: "UDP",       # User Datagram Protocol
    3478: "STUN",    # STUN, utilisé sur le port 3478 en UDP
}


# In[231]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]

# Liste pour stocker les indices des paquets pertinents
packet_indices = []

# Parcourir les paquets IP uniquement
for index, p in enumerate(ip_packets):
    control_flag = False  # Par défaut, le flag de contrôle est à False

    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_payload = bytes(p[UDP].payload)
        if udp_payload:
            control_flag = (udp_payload[0] & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC

        # Vérifie si le port source ou destination correspond au protocole STUN
        if p[UDP].sport == 3478 or p[UDP].dport == 3478:
            proto_transport_name = "STUN"  # Identifie comme STUN
        else:
            proto_transport_name = "UDP"  # Sinon, identifie comme UDP
    
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_transport_name = "TCP"

    else:
        proto_transport_name = "Unknown"

    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        packet_indices.append(index)  # Stocke l'indice du paquet pertinent

        # Ajout des données extraites dans la liste
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_transport_name,  # UDP ou STUN
            'Length': len(p),
            'Control Flag': control_flag,
        })


    index+=1
    


# On transforme maintenant nos données en dataframe pour pouvoir les analyser

# In[232]:


df_video_wifi=pd.DataFrame(p_data)
df_video_wifi['Protocol Transport'].value_counts()


# ### Temps d'Arrivée

# On nous demande alors d'analyser les temps d'arrivée des paquets de données. On crée un dataframe qui ne contient que les paquets de données. 

# In[233]:


df_video_wifi_no_control = df_video_wifi[df_video_wifi.index.isin(df_video_wifi[df_video_wifi['Control Flag'] == False].index)]


# On représente l'histogramme des temps d'arrivée

# In[234]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_wifi_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("ConvWifiArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[235]:


# Calculer les statistiques de base
data=df_video_wifi_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour la conversation par wifi')
print('Statistiques des temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('==============================================================')

# ### Longueur paquets

# In[236]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_video_wifi['Protocol Transport'].unique()
data_by_protocol = [df_video_wifi[df_video_wifi['Protocol Transport'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("taillePaquetsWifi.pdf", format="pdf")
plt.show()


# In[237]:




# Exemple de données (remplacez par vos propres données)
data = df_video_wifi['Length']

# Ajuster un modèle de mélange de Gaussiennes (2 composantes)
gmm = GaussianMixture(n_components=2)
gmm.fit(data.to_numpy().reshape(-1, 1))

# Générer des échantillons de la distribution ajustée
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
pdf_gmm = np.exp(gmm.score_samples(x))  # Probabilité d'appartenance à chaque composant

# Effectuer un test KS sur les données et la distribution ajustée
ks_statistic, p_value = stats.ks_2samp(data, x.flatten())

# Afficher la p-valeur
print(f"p-valeur du test KS : {p_value}")

# Tracer l'histogramme des données et la courbe de la distribution ajustée
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogramme des données')
plt.plot(x, pdf_gmm, label='Ajustement - Mélange de 2 Gaussiennes', color='r', linewidth=2)

# Ajouter les titres et la légende
plt.title('Histogramme et ajustement du modèle bimodal', fontsize=14)
plt.xlabel('Valeurs', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.legend()
plt.savefig("ajustement_conv_wifi.pdf", format="pdf", bbox_inches="tight")

# Afficher le graphique
plt.show()


# In[238]:


# Calculer les statistiques de base
data=df_video_wifi['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques de longueur des paquets')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('==============================================================')

# ### Analyse de la Gigue

# On ne considère ici que les paquets de données car la gigue concerne les temps d'inter-arrivée.

# In[239]:


df_video_wifi_no_control['Inter-Arrival Time']=df_video_wifi_no_control['Time'].diff().fillna(0)
df_video_wifi_no_control['Inter-Arrival Time'] = pd.to_numeric(df_video_wifi_no_control['Inter-Arrival Time'], errors='coerce')


# In[240]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_wifi_no_control['Inter-Arrival Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Inter-Arrival Time")

plt.title("Histogram of Inter-Arrival Time", fontsize=14)
plt.xlabel("Inter-Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("conv_iat_untreated_wifi.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[241]:


# Données : temps d'inter-arrivée
data = df_video_wifi_no_control['Inter-Arrival Time'].values

# Créer l'histogramme des données
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data), np.std(data)
norm_pdf = stats.norm.pdf(np.linspace(data.min(), data.max(), 1000), mu, sigma)
plt.plot(np.linspace(data.min(), data.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data)
exp_pdf = stats.expon.pdf(np.linspace(data.min(), data.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data.min(), data.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("conv_normal_and_exponential_fits_wifi.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()



# In[242]:


# Calculer les statistiques de base
data=df_video_wifi_no_control['Inter-Arrival Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
zero_percentage = (data == 0).mean() * 100  # Pourcentage de zéro
# Afficher les résultats
print('Statistiques des temps d inter-arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print(f"Pourcentage de valeurs égales à zéro: {zero_percentage:.2f}%")
print('==============================================================')

# In[243]:


df_video_wifi_no_control_no0 = df_video_wifi_no_control[df_video_wifi_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_video_wifi_no_control_no0[df_video_wifi_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("conv_normal_and_exponential_fits_filtered_wifi.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()




# ## Conversation Teams en ethernet

# In[244]:


pcap_file = r'ConvEthernet.pcapng'
data=rdpcap(pcap_file)


# In[245]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]

# Liste pour stocker les indices des paquets pertinents
packet_indices = []

# Parcourir les paquets IP uniquement
for index, p in enumerate(ip_packets):
    control_flag = False  # Par défaut, le flag de contrôle est à False

    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_payload = bytes(p[UDP].payload)
        if udp_payload:
            control_flag = (udp_payload[0] & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC

        # Vérifie si le port source ou destination correspond au protocole STUN
        if p[UDP].sport == 3478 or p[UDP].dport == 3478:
            proto_transport_name = "STUN"  # Identifie comme STUN
        else:
            proto_transport_name = "UDP"  # Sinon, identifie comme UDP
    
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_transport_name = "TCP"

    else:
        proto_transport_name = "Unknown"

    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        packet_indices.append(index)  # Stocke l'indice du paquet pertinent

        # Ajout des données extraites dans la liste
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_transport_name,  # UDP ou STUN
            'Length': len(p),
            'Control Flag': control_flag,
        })


    index+=1


# In[246]:


df_video_ethernet=pd.DataFrame(p_data)
df_video_ethernet


# ### Arrivées

# In[247]:


df_video_ethernet_no_control = df_video_ethernet[df_video_ethernet.index.isin(df_video_ethernet[df_video_ethernet['Control Flag'] == False].index)]


# In[248]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_ethernet_no_control['Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Arrival Time")

plt.title("Histogram of Arrival Time", fontsize=14)
plt.xlabel("Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("ConvEthernetArrivée.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[249]:


# Calculer les statistiques de base
data=df_video_ethernet_no_control['Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques pour la conversation par ethernet')
print('Statistiques des temps d arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('==============================================================')

# ### Longueur des paquets

# In[250]:


# Créer un histogramme des valeurs de 'length'
plt.figure(figsize=(12, 6))


# Créer un histogramme par protocole et les empiler
protocols = df_video_ethernet['Protocol Transport'].unique()
data_by_protocol = [df_video_ethernet[df_video_ethernet['Protocol Transport'] == protocol]['Length'] for protocol in protocols]

# Placer l'histogramme empilé avec différentes couleurs par protocole
plt.hist(data_by_protocol, bins=50, stacked=True, label=protocols, alpha=0.7)

#plt.hist(df_wifi['Length'], bins=100, color='blue', alpha=0.7)  # Ajustez le nombre de bins selon la distribution
plt.title('Histogramme empilé des tailles de paquets IP par protocole')
plt.xlabel('Taille des paquets (octets)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.legend(title="Protocoles")
plt.savefig("taillePaquetsEthernet.pdf", format="pdf")
plt.show()


# In[251]:



# Exemple de données (remplacez par vos propres données)
data = df_video_ethernet['Length']

# Ajuster un modèle de mélange de Gaussiennes (2 composantes)
gmm = GaussianMixture(n_components=2)
gmm.fit(data.to_numpy().reshape(-1, 1))

# Générer des échantillons de la distribution ajustée
x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
pdf_gmm = np.exp(gmm.score_samples(x))  # Probabilité d'appartenance à chaque composant

# Effectuer un test KS sur les données et la distribution ajustée
ks_statistic, p_value = stats.ks_2samp(data, x.flatten())

# Afficher la p-valeur
print(f"p-valeur du test KS : {p_value}")

# Tracer l'histogramme des données et la courbe de la distribution ajustée
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histogramme des données')
plt.plot(x, pdf_gmm, label='Ajustement - Mélange de 2 Gaussiennes', color='r', linewidth=2)

# Ajouter les titres et la légende
plt.title('Histogramme et ajustement du modèle bimodal', fontsize=14)
plt.xlabel('Valeurs', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.legend()
plt.savefig("ajustement_conv_eth.pdf", format="pdf", bbox_inches="tight")

# Afficher le graphique
plt.show()


# In[252]:


# Calculer les statistiques de base
data=df_video_ethernet['Length']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
# Afficher les résultats
print('Statistiques des longueurs de paquets')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print('==============================================================')

# In[253]:


df_video_ethernet_no_control['Inter-Arrival Time']=df_video_ethernet_no_control['Time'].diff().fillna(0)
df_video_ethernet_no_control['Inter-Arrival Time'] = pd.to_numeric(df_video_ethernet_no_control['Inter-Arrival Time'], errors='coerce')


# In[254]:


plt.figure(figsize=(12, 6))
plt.hist(df_video_ethernet_no_control['Inter-Arrival Time'], bins=100, color='blue', alpha=0.7, label="Histogram of Inter-Arrival Time")

plt.title("Histogram of Inter-Arrival Time", fontsize=14)
plt.xlabel("Inter-Arrival Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend(fontsize=12)
plt.savefig("conv_iat_untreated_eth.pdf", format="pdf", bbox_inches="tight")
plt.show()


# In[255]:



# Données : temps d'inter-arrivée
data = df_video_ethernet_no_control['Inter-Arrival Time'].values

# Créer l'histogramme des données
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data), np.std(data)
norm_pdf = stats.norm.pdf(np.linspace(data.min(), data.max(), 1000), mu, sigma)
plt.plot(np.linspace(data.min(), data.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data)
exp_pdf = stats.expon.pdf(np.linspace(data.min(), data.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data.min(), data.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("conv_normal_and_exponential_fits_eth.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()



# In[256]:


# Calculer les statistiques de base
data=df_video_ethernet_no_control['Inter-Arrival Time']
min_time = data.min()  # Minimum
max_time = data.max()  # Maximum
median_time = data.median()  # Médiane
quantiles = data.quantile([0.25, 0.5, 0.75])  # Quartiles
mean_time = data.mean()  # Premier moment: Moyenne
second_moment = np.mean(data**2)  # Deuxième moment: Variance
variance=data.var()
zero_percentage = (data == 0).mean() * 100  # Pourcentage de zéro
# Afficher les résultats
print('Statistiques des temps d inter-arrivée')
print(f"Premier moment (moyenne): {mean_time}")
print(f"Deuxième moment : {second_moment}")
print(f"Variance : {variance}")
# Afficher les résultats
print(f"Minimum: {min_time}")
print(f"Maximum: {max_time}")
print(f"Médiane: {median_time}")
print(f"Quartiles (25%, 50%, 75%): {quantiles}")
print(f"Pourcentage de valeurs égales à zéro: {zero_percentage:.2f}%")


# In[257]:


df_video_ethernet_no_control_no0 = df_video_ethernet_no_control[df_video_ethernet_no_control['Inter-Arrival Time'] != 0]
# Filtrer les données pour ne retenir que les temps d'inter-arrivée < 0.1 s
data_filtered = df_video_ethernet_no_control_no0[df_video_ethernet_no_control_no0['Inter-Arrival Time'] < 0.1]['Inter-Arrival Time'].values

# Créer l'histogramme des données filtrées
plt.figure(figsize=(12, 8))

# Subplot 1 : Ajustement à une distribution normale
plt.subplot(1, 2, 1)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
mu, sigma = np.mean(data_filtered), np.std(data_filtered)
norm_pdf = stats.norm.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), mu, sigma)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), norm_pdf, label=f"Normal Fit (μ={mu:.4f}, σ={sigma:.4f})", color='green', linewidth=2)
plt.title("Normal Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement normal via le test KS
ks_norm_stat, ks_norm_pval = stats.kstest(data_filtered, 'norm', args=(mu, sigma))
plt.text(0.05, 0.95, f"KS p-value: {ks_norm_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Subplot 2 : Ajustement à une distribution exponentielle
plt.subplot(1, 2, 2)
plt.hist(data_filtered, bins=100, density=True, alpha=0.7, color='blue', label="Data Histogram")
lambda_exp = 1 / np.mean(data_filtered)
exp_pdf = stats.expon.pdf(np.linspace(data_filtered.min(), data_filtered.max(), 1000), scale=1/lambda_exp)
plt.plot(np.linspace(data_filtered.min(), data_filtered.max(), 1000), exp_pdf, label=f"Exponential Fit (λ={lambda_exp:.4f})", color='red', linewidth=2)
plt.title("Exponential Distribution Fit")
plt.xlabel("Inter-Arrival Time (s)")
plt.ylabel("Density")
plt.legend()

# Calcul de la p-value pour l'ajustement exponentiel via le test KS
ks_exp_stat, ks_exp_pval = stats.kstest(data_filtered, 'expon', args=(0, 1/lambda_exp))
plt.text(0.05, 0.95, f"KS p-value: {ks_exp_pval:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Ajuster les marges et afficher
plt.tight_layout()

# Sauvegarder le graphique en PDF
plt.savefig("conv_normal_and_exponential_fits_filtered_eth.pdf", format="pdf", bbox_inches="tight")

# Afficher les sous-graphiques
plt.show()
##########################################################################################
print('Partie Machine Learning')
#######################################################################################

protocol_names = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
}


# In[227]:


def get_app_protocol_video_udp(source_port, destination_port):
    if (source_port == 443 or destination_port == 443):
        return 'Quic'
    elif (source_port == 53 or destination_port == 53):
        return 'DNS' #Zoom Custom Protocol
    else:
        return 'xUDP'## On retira ces paquets non identifiés du set de données
def get_app_protocol_video_tcp(source_port,destination_port,length):
    if (source_port == 443 or destination_port ==443):
        return 'HTTPS'
    elif (source_port == 7680 or destination_port ==7680):
        if length==70:
            return 'ICMP'
        else:
            return 'xTCP'
    else:
        return 'xTCP'


# ### Traitement et nettoyage du set de données issu de la vidéo youtube par Wifi

# In[228]:


pcap_file = r'VideoWifi.pcapng'
data=rdpcap(pcap_file)


# In[229]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_video_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_video_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[230]:


df_video_wifi=pd.DataFrame(p_data)
df_video_wifi = df_video_wifi[df_video_wifi['Protocol Application'] != False]
df_video_wifi['Liaison'] = 'Wifi'
df_video_wifi['Expérience'] = 'Vidéo'
df_video_wifi['Inter-Arrival Time']=df_video_wifi['Time'].diff().fillna(0.)
df_video_wifi['Control Flag'].value_counts()


# ### Traitement et nettoyade de la vidéo par éthernet

# In[231]:


pcap_file = r'VideoEthernet.pcapng'
data=rdpcap(pcap_file)


# In[232]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_video_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_video_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[233]:


df_video_ethernet=pd.DataFrame(p_data)
df_video_ethernet = df_video_ethernet[df_video_ethernet['Protocol Application'] != False]
df_video_ethernet['Liaison'] = 'Ethernet'
df_video_ethernet['Expérience'] = 'Vidéo'
df_video_ethernet['Inter-Arrival Time']=df_video_ethernet['Time'].diff().fillna(0.)
df_video_ethernet['Control Flag'].value_counts()


# ### Traitement et nettoyage de la voix sur IP

# In[234]:


def get_app_protocol_voix_udp(source_port, destination_port):
    if (source_port == 3480 or destination_port == 3480):
        return 'STUN'
    elif (source_port == 443 or destination_port == 443):
        return 'Quic'
    elif (source_port == 53 or destination_port == 53):
        return 'DNS' #Zoom Custom Protocol
    else:
        return 'xUDP'## On retira ces paquets non identifiés du set de données
def get_app_protocol_voix_tcp(source_port,destination_port,length):
    if (source_port == 443 or destination_port ==443):
        return 'HTTPS'
    elif (source_port == 7680 or destination_port ==7680):
        if length==70:
            return 'ICMP'
        else:
            return 'xTCP'
    else:
        return 'xTCP'


# In[235]:


pcap_file = r'ConvWifi.pcapng'
data=rdpcap(pcap_file)


# In[236]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_voix_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_voix_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[237]:


df_voix_wifi=pd.DataFrame(p_data)
df_voix_wifi = df_voix_wifi[df_voix_wifi['Protocol Application'] != False]
df_voix_wifi['Liaison'] = 'Wifi'
df_voix_wifi['Expérience'] = 'Voix'
df_voix_wifi['Inter-Arrival Time']=df_voix_wifi['Time'].diff().fillna(0.)
df_voix_wifi['Control Flag'].value_counts()


# ### Traitement et nettoyage de la voix sur ip par ethernet

# In[ ]:


pcap_file = r'ConvEthernet.pcapng'
data=rdpcap(pcap_file)


# In[ ]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_voix_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_voix_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[ ]:


df_voix_ethernet=pd.DataFrame(p_data)
df_voix_ethernet = df_voix_ethernet[df_voix_ethernet['Protocol Application'] != False]
df_voix_ethernet['Liaison'] = 'Ethernet'
df_voix_ethernet['Expérience'] = 'Voix'
df_voix_ethernet['Inter-Arrival Time']=df_voix_ethernet['Time'].diff().fillna(0.)
df_voix_ethernet['Control Flag'].value_counts()


# ### Traitement et nettoyage des téléchargements

# In[ ]:


def get_app_protocol_download_udp(source_port, destination_port):
    if (source_port == 3480 or destination_port == 3480):
        return 'STUN'
    elif (source_port == 443 or destination_port == 443):
        return 'Quic'
    elif (source_port == 53 or destination_port == 53):
        return 'DNS' #Zoom Custom Protocol
    else:
        return 'xUDP'## On retira ces paquets non identifiés du set de données
def get_app_protocol_download_tcp(source_port,destination_port,length):
    if (source_port == 443 or destination_port ==443):
        return 'HTTPS'
    elif (source_port == 7680 or destination_port ==7680):
        if length==70:
            return 'ICMP'
        else:
            return 'xTCP'
    elif (source_port == 80 or destination_port ==80):
        return 'HTTP'
    else:
        return 'xTCP'


# In[ ]:


pcap_file = r'DownloadWifi.pcapng'
data=rdpcap(pcap_file)


# In[ ]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_download_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_download_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[ ]:


df_download_wifi=pd.DataFrame(p_data)
df_download_wifi = df_download_wifi[df_download_wifi['Protocol Application'] != False]
df_download_wifi['Liaison'] = 'Wifi'
df_download_wifi['Expérience'] = 'Download'
df_download_wifi['Inter-Arrival Time']=df_download_wifi['Time'].diff().fillna(0.)
df_download_wifi['Control Flag'].value_counts()


# ### Traitement et nettoyage des données de téléchargement par Ethernet

# In[ ]:


pcap_file = r'DownloadEthernet.pcapng'
data=rdpcap(pcap_file)


# In[ ]:


# Liste pour stocker les données extraites
p_data = []

# Temps de début pour normaliser les timestamps
start_time = data[0].time if len(data) > 0 else 0

# Filtrer les paquets pour ne garder que ceux contenant une couche IP
ip_packets = [p for p in data if IP in p]
list = []
# Parcourir les paquets IP uniquement
for p in ip_packets:
    # Vérifie si c'est un paquet UDP
    if UDP in p:
        udp_pckt = bytes(p[UDP].payload)[0]
        control_flag = (udp_pckt & 0xC0) == 0xC0  # Basé sur la documentation du protocole QUIC
        proto_couche_sup=get_app_protocol_download_udp(p[IP].payload.sport, p[IP].payload.dport)
    # Vérifie si c'est un paquet TCP
    elif TCP in p and hasattr(p[TCP], 'flags'):
        control_flag = TCP_control(p)
        proto_couche_sup=get_app_protocol_download_tcp(p[IP].payload.sport, p[IP].payload.dport,len(p))
    else:
        proto_couche_cup='?'
        control_flag = False
    # Vérifie si la charge utile IP a des ports source et destination
    # Vérifie si la charge utile IP a des ports source et destination
    if hasattr(p[IP].payload, 'sport') and hasattr(p[IP].payload, 'dport'):
        proto_number = p[IP].proto  # Numéro de protocole
        proto_name = protocol_names.get(proto_number, f"Unknown ({proto_number})")  # Nom du protocole
        p_data.append({
            'Time': p.time - start_time,
            'Source IP': p[IP].src,
            'Destination IP': p[IP].dst,
            'Source Port': p[IP].payload.sport,
            'Destination Port': p[IP].payload.dport,
            'Protocol Transport': proto_name,  # Utilisation du nom ici
            'Protocol Application':proto_couche_sup,
            'Length': len(p),
            'Control Flag': control_flag,
        })
    


# In[ ]:
print('Rassemblement des datasets...')

df_download_ethernet=pd.DataFrame(p_data)
df_download_ethernet = df_download_ethernet[df_download_ethernet['Protocol Application'] != False]
df_download_ethernet['Liaison'] = 'Ethernet'
df_download_ethernet['Expérience'] = 'Download'
df_download_ethernet['Inter-Arrival Time']=df_download_ethernet['Time'].diff().fillna(0.)
df_download_ethernet['Control Flag'].value_counts()


# ## Fusion des datasets
# 

# In[ ]:


df1 = pd.concat([df_video_wifi, df_video_ethernet], ignore_index=False)
df2 = pd.concat([df_voix_wifi, df_voix_ethernet], ignore_index=False)
df3 = pd.concat([df_download_wifi, df_download_ethernet], ignore_index=False)
df= pd.concat([df1, df2,df3], ignore_index=True)
df=df.drop(columns=['Time'], axis='columns')#La colonne Time n'a plus de sens car on a mélangé toutes les expériences


# Le but des algorthmes de machine learning est de pouvoir déterminer certaines variables en fonction des attributs.

# ## Prédiction des paquets de contrôle

# On doit d'abord déterminer quels attributs utiliser ou non pour prédire au mieux si nous sommes face à un paquet de contrôle ou non. 

# In[ ]:
print('Traitement des données...')

df['Source IP numérique']=df['Source IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
df['Destination IP numérique']=df['Destination IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
encoder = LabelEncoder()
df['Protocole Transport numérique']=encoder.fit_transform(df['Protocol Transport'])
df['Protocole Application numérique']=encoder.fit_transform(df['Protocol Application'])
df


# In[ ]:
################################################################################
print('#########################################################################')
print('Prédiction des paquets de contrôle...')
X=df.drop(columns=['Control Flag','Source IP', 'Destination IP','Liaison','Expérience', 'Protocol Transport', 'Protocol Application'], axis='columns')
y=df['Control Flag'].apply(int)
X


# In[ ]:


# Créer le dataframe W en ajoutant y à X
W = X.copy()  # Copie de X
W['Control Flag'] = y  # Ajouter y sous la colonne 'Control Flag'

# Calculer la matrice de corrélation de W
correlation_matrix = W.corr()

# Créer une carte thermique de la matrice de corrélation
plt.figure(figsize=(10, 8))  # Taille de la figure
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Ajouter un titre
plt.title("Matrice de Corrélation des Variables")

# Sauvegarder en PDF
plt.savefig("correlation_matrix_ControlFlag.pdf", format="pdf")

# Afficher le graphique
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

Regressor = LogisticRegression()
Regressor.fit(X_train, y_train)
y_pred = Regressor.predict(X_test)

# Calcul de la MSE pour le modèle de régression logistique
mse_regressor = mean_squared_error(y_test, y_pred)
print("Précision du modèle (MSE) - Régression Logistique :", mse_regressor)

# Modèle de base (naïf) : prédiction avec la moyenne de y_train
y_base_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=np.float64)
y_base_pred = np.round(y_base_pred)  # Arrondir les prédictions pour avoir des valeurs entières si nécessaire

# Calcul de la MSE pour le modèle de base
mse_base = mean_squared_error(y_test, y_base_pred)
print("Précision du modèle (MSE) - Modèle de Base :", mse_base)

# Affichage des résultats de la matrice de confusion pour le modèle de régression logistique
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Remplacer y_pred par les prédictions du modèle
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.savefig("confusion_matrix_regression_control.pdf", format="pdf")

# Classification Report pour le modèle de régression logistique
print("\nClassification Report - Régression Logistique :")
print(classification_report(y_test, y_pred, labels=[0, 1],zero_division=0))

# Classification Report pour le modèle de base
print("\nClassification Report - Modèle de Base :")
print(classification_report(y_test, y_base_pred, labels=[0, 1],zero_division=0))


# On voit qu'un modèle simple de régression linéaire montre de très bonnes performances. Cependant, on remaque que le modèle possède beaucoup d'attributs. Il est donc pertinent d'entamer une recherche afin de savoir quels attriubuts sont les plus significatifs pour faire une bonne prévision. 

# In[ ]:
print('===============================================================')
print('Prédiction avec moins d attributs')
X_small=X.drop(['Destination IP numérique','Source IP numérique','Protocole Transport numérique','Source Port','Length','Inter-Arrival Time'], axis='columns')
X_small


# In[ ]:


# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# Modèle de régression logistique
Regressor = LogisticRegression()
Regressor.fit(X_train, y_train)
y_pred = Regressor.predict(X_test)

# Calcul de la MSE pour le modèle de régression logistique
mse_regressor = mean_squared_error(y_test, y_pred)
print("Précision du modèle (MSE) - Régression Logistique :", mse_regressor)

# Modèle de base (naïf) : prédiction avec la moyenne de y_train
y_base_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=np.float64)
y_base_pred = np.round(y_base_pred)  # Arrondir les prédictions pour avoir des valeurs entières si nécessaire

# Calcul de la MSE pour le modèle de base
mse_base = mean_squared_error(y_test, y_base_pred)
print("Précision du modèle (MSE) - Modèle de Base :", mse_base)

# Affichage des résultats de la matrice de confusion pour le modèle de régression logistique
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Remplacer y_pred par les prédictions du modèle
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()

# Classification Report pour le modèle de régression logistique
print("\nClassification Report - Régression Logistique :")
print(classification_report(y_test, y_pred, labels=[0, 1],zero_division=0))

# Classification Report pour le modèle de base
print("\nClassification Report - Modèle de Base :")
print(classification_report(y_test, y_base_pred, labels=[0, 1],zero_division=0))


# On voit qu'on a une bonne performance en prenant l'adresse Ip de la destination, en effet, le modèle a tendance alors à classer tous les paquets arrivant comme des paquets de données et repartant comme des paquets de contrôle. Sans cela, la source de l'adresse IP joue également un rôle significatif. On a aussi une très bonne estimation grâce aux ports destinations et protocoles d'application. On a une précision moyenne avec juste la longueur des paquets. Ce modèle suggère que les paquets les plus courts sont considérés comme paquets de contrôle. 

# ## Prédiction du protocole d'application

# ### Regression

# In[ ]:
print('#######################################################################')
print('Prédiction du modèle d application')

X=df.drop(columns=['Source IP', 'Destination IP','Liaison','Expérience', 'Protocol Transport', 'Protocol Application', 'Protocole Application numérique', 'Source Port','Destination Port'], axis='columns')
y=df['Protocole Application numérique'].apply(int)
y.value_counts()
X


# In[ ]:


# Créer le dataframe W en ajoutant y à X
W = X.copy()  # Copie de X
W['Protocole Application numérique'] = y  # Ajouter y sous la colonne 'Control Flag'

# Calculer la matrice de corrélation de W
correlation_matrix = W.corr()

# Créer une carte thermique de la matrice de corrélation
plt.figure(figsize=(10, 8))  # Taille de la figure
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Ajouter un titre
plt.title("Matrice de Corrélation des Variables")

# Sauvegarder en PDF
plt.savefig("correlation_matrix_Application.pdf", format="pdf")

# Afficher le graphique
plt.show()


# In[ ]:


# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# Modèle de classification : Arbre de décision
clf = DecisionTreeClassifier(class_weight='balanced', random_state=22)#class_weight qui ajuste la fonction de perte en fonction de l'importance relative des classes, ici le calcul est fait automatiquement
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Modèle de base (naïf) : prédiction avec la classe majoritaire de y_train
y_base_pred = np.full_like(y_test, fill_value=np.bincount(y_train).argmax(), dtype=np.int64)

y_pred_best=y_pred

# Affichage des résultats de la matrice de confusion pour le modèle d'arbre de décision
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusionMatrixDtApplication.pdf", format='pdf')

# Classification Report pour le modèle d'arbre de décision
print("\nClassification Report - Arbre de Décision :")
print(classification_report(y_test, y_pred,zero_division=0))

# Classification Report pour le modèle de base
print("\nClassification Report - Modèle de Base :")
print(classification_report(y_test, y_base_pred,zero_division=0))

print('======================================================================')
# On veut également ici essayer de dégager les attributs les plus significatifs

# In[ ]:
print('Prédiction avec moins d attributs')

X_small=X[['Length', 'Inter-Arrival Time']]


# In[ ]:


# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_small, y, test_size=0.3, random_state=22)

# Modèle de classification : Arbre de décision
clf = DecisionTreeClassifier(class_weight='balanced',random_state=22)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Affichage des résultats de la matrice de confusion pour le modèle d'arbre de décision
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusionMatrixDtApplicationMin.pdf", format='pdf')

# Classification Report pour le modèle d'arbre de décision
print("\nClassification Report - Arbre de Décision :")
print(classification_report(y_test, y_pred,zero_division=0))

# Classification Report pour le modèle de base
print("\nClassification Report - Modèle avec toutes features :")
print(classification_report(y_test, y_pred_best,zero_division=0))


# On voit que la longueur joue un rôle significatif tout comme le temps d'inter-arrivée. Pour la suite, nous décidons de ne pas reprendre le port source et destination comme nous utilisons deja cette information pour déduire directement le protocole d'application, ça serait trop facile. Le protocole de transport permet aussi d'améliorer les scores, mets nous allons décider de ne pas le prendre car certains protocoles sont toujours éliminés pour un certain protocole de transport. Nous allons donc essayer d'optimiser la prédiction en ne prenant que 'Inter-Arrival Time' et 'Length' comme features.
# 

# In[ ]:


# Définir la grille des hyperparamètres à tester
param_grid = {
    'max_depth': [3, 5, 10, 15, None],  # Profondeur maximale
    'min_samples_split': [2, 5, 10],   # Minimum d'échantillons pour un split
    'min_samples_leaf': [1, 2, 5],     # Minimum d'échantillons par feuille
    'class_weight': [None, 'balanced'] # Équilibrage des classes
}

# Initialisation du modèle
clf = DecisionTreeClassifier(random_state=22)

# Configuration du GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)

# Exécution de la recherche
grid_search.fit(X_train, y_train)

# Résultats
print("Meilleurs paramètres trouvés :", grid_search.best_params_)
print("Meilleur score F1 (macro) :", grid_search.best_score_)

# Modèle optimisé
best_model = grid_search.best_estimator_

# Prédictions avec le modèle optimisé
y_pred_optimized = best_model.predict(X_test)

# Affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred_optimized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot()
plt.savefig("confusionMatrixDtApplicationMinOptimized.pdf", format='pdf')

# Classification report pour le modèle optimisé
print("\nClassification Report - Modèle Optimisé :")
print(classification_report(y_test, y_pred_optimized,zero_division=0))

# Calcul de l'accuracy du modèle optimisé
accuracy_optimized = best_model.score(X_test, y_test)
print(f"Précision (Accuracy) du modèle optimisé : {accuracy_optimized:.4f}")


# In[ ]:


clf = DecisionTreeClassifier(random_state=22)

# Calculer la courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='f1_macro', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

# Calcul de la moyenne et de l'écart-type des scores pour chaque taille d'échantillon
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Tracer la courbe d'apprentissage
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label="Score d'entraînement", color='blue')
plt.plot(train_sizes, test_scores_mean, label="Score de test", color='red')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='red')

# Ajouter des labels et un titre
plt.title("Courbe d'apprentissage du modèle")
plt.xlabel("Nombre d'échantillons d'entraînement")
plt.ylabel("Score F1 (macro)")
plt.legend()
plt.savefig('score.pdf', format='pdf')

# Afficher le graphique
plt.show()


# On peut voir qu'on a maintenant une précision légèrement meilleure par rapport à avant (augmentation du f1-score) pour les paramètres trouvés grâce au grid search.
# 

# ## Estimation du moyen de liaison 

# on retire les sources IP et destination IP car les adresses réseaux permettent de déterminer le moyen de liaison de manière déterministe. 

# In[ ]:
print('##################################################################')
print('Prédiction du moyen de liaison')

X=df.drop(columns=['Liaison','Expérience', 'Protocol Transport', 'Protocol Application','Source IP', 'Destination IP', 'Source IP numérique', 'Destination IP numérique'], axis='columns')
df['Liaison'] = df['Liaison'].map({'Wifi': 1, 'Ethernet': 0})
y=df['Liaison'].apply(int)
y.value_counts()
X


# In[ ]:


# Créer le dataframe W en ajoutant y à X
W = X.copy()  # Copie de X
W['Liaison'] = y  # Ajouter y sous la colonne 'Control Flag'

# Calculer la matrice de corrélation de W
correlation_matrix = W.corr()

# Créer une carte thermique de la matrice de corrélation
plt.figure(figsize=(10, 8))  # Taille de la figure
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Ajouter un titre
plt.title("Matrice de Corrélation des Variables")

# Sauvegarder en PDF
plt.savefig("correlation_matrix_Liaison.pdf", format="pdf")

# Afficher le graphique
plt.show()


# In[ ]:


# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# Modèle de régression logistique
Regressor = LogisticRegression()
Regressor.fit(X_train, y_train)
y_pred = Regressor.predict(X_test)

# Calcul de la MSE pour le modèle de régression logistique
mse_regressor = mean_squared_error(y_test, y_pred)
print("Précision du modèle (MSE) - Régression Logistique :", mse_regressor)


# Affichage des résultats de la matrice de confusion pour le modèle de régression logistique
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Remplacer y_pred par les prédictions du modèle
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.savefig("confusionMatrixLiaisonRegression.pdf", format='pdf')

# Classification Report pour le modèle de régression logistique
print("\nClassification Report - Régression Logistique :")
print(classification_report(y_test, y_pred, labels=[0, 1],zero_division=0))



# In[ ]:



# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

# Modèle K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)  # Choisir un k par défaut, vous pouvez optimiser ce paramètre
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Affichage des résultats pour KNN
print("=== K-Nearest Neighbors (KNN) ===")
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=[0, 1])
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=[0, 1])
disp_knn.plot()
plt.savefig("confusionMatrixLiaisonKnn.pdf", format='pdf')
print("\nClassification Report - KNN :")
print(classification_report(y_test, y_pred_knn, labels=[0, 1],zero_division=0))

# Modèle Decision Tree
dt = DecisionTreeClassifier(random_state=22)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Affichage des résultats pour Decision Tree
print("\n=== Decision Tree ===")
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=[0, 1])
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=[0, 1])
disp_dt.plot()
plt.savefig("confusionMatrixLiaisonDt.pdf", format='pdf')
print("\nClassification Report - Decision Tree :")
print(classification_report(y_test, y_pred_dt, labels=[0, 1],zero_division=0))
X


# Les performances sont meilleures avec ces algorithmes. On garde le decision tree qui a une performance légèrement supérieure. Maintenant nous allons déterminer les features importantes pour ethernet

# In[ ]:

print('=======================================================')
print('Prédiction avec moins d attributs')
X_small=X[['Destination Port','Source Port']]


# In[ ]:

# Division des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_small, y, test_size=0.3, random_state=22)

# Modèle Decision Tree
dt = DecisionTreeClassifier(random_state=22)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Affichage des résultats pour Decision Tree
print("\n=== Decision Tree ===")
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=[0, 1])
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=[0, 1])
disp_dt.plot()
plt.savefig("confusionMatrixLiaisonDtMin.pdf", format='pdf')
print("\nClassification Report - Decision Tree :")
print(classification_report(y_test, y_pred_dt, labels=[0, 1],zero_division=0))


# Destination port et source port semble aider participer pour la majorité aux bons résultats pour déterminer le moyen de liaison. Nous retenons donc ceux-ci


