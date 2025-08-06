import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from anfis import anfis, membership

# --- 1. Memuat Dataset dan Ekstraksi Fitur ---
print("1. Memuat dan Memproses Dataset...")

# Pastikan nama file dan jalurnya sudah benar
X_asli = np.load('/home/pi/MyEnv/anfis_extracted_features/anfis_features_50k_belakang.npy')
Y_target = np.load('/home/pi/MyEnv/anfis_extracted_features/anfis_labels_50k_belakang.npy')
# --- SAMPAI DI SINI ---

print(f"   Dataset asli dimuat: {X_asli.shape[0]} sampel dengan {X_asli.shape[1]} fitur.")

# --- 2. Reduksi Dimensi dengan PCA ---
print("2. Menerapkan PCA untuk Reduksi Dimensi...")
jumlah_fitur_pca = 10 

pca = PCA(n_components=jumlah_fitur_pca)
X_pca = pca.fit_transform(X_asli)

print(f"   Data berhasil direduksi menjadi: {X_pca.shape[0]} sampel dengan {X_pca.shape[1]} fitur.")
# --- BARIS INI BISA ANDA TAMBAHKAN UNTUK MELIHAT RENTANG DATA ---
print("\n--- Statistik Data Fitur Setelah PCA ---")
for i in range(X_pca.shape[1]):
    fitur_data = X_pca[:, i]
    print(f"   Fitur {i+1}:")
    print(f"     Min:  {np.min(fitur_data):.2f}")
    print(f"     Max:  {np.max(fitur_data):.2f}")
    print(f"     Mean: {np.mean(fitur_data):.2f}")
print("--------------------------------------\n")

# --- 3. Mengkonfigurasi dan Melatih Model ANFIS ---
print("3. Mengkonfigurasi dan Melatih Model ANFIS...")

# Menggunakan seluruh 98 sampel untuk pelatihan.
X_train_anfis = X_pca[:, :3]
Y_train_anfis = Y_target

# Menggunakan 3 fungsi keanggotaan per fitur
# Sesuaikan mean dan sigma agar cakupannya lebih luas dan lebih banyak
mf = [
    # Fitur Input 1 (Rentang: -4908.84 hingga 2431.14)
    [
        ['gaussmf', {'mean': -4000., 'sigma': 1500.}],
        ['gaussmf', {'mean': 0., 'sigma': 1500.}],
        ['gaussmf', {'mean': 2000., 'sigma': 1000.}]
    ],
    # Fitur Input 2 (Rentang: -950.33 hingga 2722.75)
    [
        ['gaussmf', {'mean': -800., 'sigma': 500.}],
        ['gaussmf', {'mean': 0., 'sigma': 500.}],
        ['gaussmf', {'mean': 1500., 'sigma': 800.}]
    ],
    # Fitur Input 3 (Rentang: -397.40 hingga 773.29)
    [
        ['gaussmf', {'mean': -300., 'sigma': 200.}],
        ['gaussmf', {'mean': 0., 'sigma': 200.}],
        ['gaussmf', {'mean': 500., 'sigma': 300.}]
    ]
]

mf_object = membership.membershipfunction.MemFuncs(mf)
anfis_model = anfis.ANFIS(X_train_anfis, Y_train_anfis, mf_object)

# Latih model selama 100 epochs
anfis_model.trainHybridJangOffLine(epochs=100)

print("   Pelatihan selesai.")

# --- 4. Plot Hasil ---
anfis_model.plotResults()
plt.show()

# --- 5. Evaluasi Hasil (opsional) ---
print(f"   Error pelatihan terakhir: {anfis_model.errors[-1]}")

