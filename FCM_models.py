import numpy as np
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Muat Dataset dan Pisahkan ---
X_gabungan = np.load('anfis_extracted_features/anfis_features_50k_depan.npy')
Y_labels = np.load('anfis_extracted_features/anfis_labels_50k_depan.npy')

#X_gabungan = np.load('anfis_extracted_features/anfis_features_50k_depan.npy')
#Y_labels = np.load('anfis_extracted_features/anfis_labels_50k_depan.npy')

'''X_gabungan = np.load('anfis_extracted_features/anfis_features_100k_belakang.npy')
Y_labels = np.load('anfis_extracted_features/anfis_labels_100k_belakang.npy')'''

#X_gabungan = np.load('anfis_extracted_features/anfis_features_100k_depan.npy')
#Y_labels = np.load('anfis_extracted_features/anfis_labels_100k_depan.npy')

X_asli = X_gabungan[Y_labels == 1]
X_palsu = X_gabungan[Y_labels == 0]

X_train_asli, X_test_asli, _, _ = train_test_split(X_asli, X_asli, test_size=0.2, random_state=42)

print(f"Dataset berhasil dipisahkan:")
print(f"  - Sampel uang asli: {len(X_asli)} sampel")
print(f"  - Sampel uang palsu: {len(X_palsu)} sampel")
print(f"  - Sampel asli untuk training: {len(X_train_asli)} sampel")
print(f"  - Sampel asli untuk testing: {len(X_test_asli)} sampel")
print("-" * 50)

# --- 2. Lakukan Standard Scaling ---
scaler = StandardScaler()
X_train_asli_scaled = scaler.fit_transform(X_train_asli)
X_test_asli_scaled = scaler.transform(X_test_asli)
X_palsu_scaled = scaler.transform(X_palsu)
print("Data berhasil diskalakan.")
print("-" * 50)


# --- 3. Pelatihan Model Fuzzy C-Means (FCM) ---
# n_clusters: jumlah kelompok yang ingin dicari
# m: parameter fuzziness. Nilai 2 adalah standar.
n_clusters = 3
m = 2

# Transpose data agar sesuai dengan format yang dibutuhkan skfuzzy
fp_data = X_train_asli_scaled.T

# Jalankan FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
    fp_data, n_clusters, m, error=0.005, maxiter=1000, init=None
)
print(f"Fuzzy C-Means berhasil dijalankan dengan {n_clusters} clusters.")


# --- 4. Tentukan Batas Anomali (Threshold) ---
# Hitung jarak dari setiap sampel ke pusat cluster terdekat
jarak_ke_pusat = np.zeros((X_train_asli_scaled.shape[0], n_clusters))
for i in range(X_train_asli_scaled.shape[0]):
    for j in range(n_clusters):
        jarak_ke_pusat[i, j] = np.linalg.norm(X_train_asli_scaled[i] - cntr[j])

jarak_minimal = np.min(jarak_ke_pusat, axis=1)

# Tentukan threshold berdasarkan persentil ke-95
threshold = np.percentile(jarak_minimal, 98) # Gunakan persentil yang sama dengan K-Means
print(f"Threshold anomali ditentukan pada jarak: {threshold:.2f}")
print("-" * 50)

# --- Untuk Dataset 50k_tampak_belakang ---
'''joblib.dump(cntr, 'fcm_cluster_centers_50k_belakang.joblib')
joblib.dump(scaler, 'fcm_scaler_50k_belakang.joblib')
joblib.dump(threshold, 'fcm_threshold_50k_belakang.joblib')
print("Model, scaler, dan threshold berhasil disimpan.")
print("-" * 50)'''

# --- Untuk Dataset 50k_tampak_depan ---
# Pastikan Anda sudah melatih model dan scaler dengan dataset yang baru
'''joblib.dump(cntr, 'fcm_cluster_centers_50k_depan.joblib')
joblib.dump(scaler, 'fcm_scaler_50k_depan.joblib')
joblib.dump(threshold, 'fcm_threshold_50k_depan.joblib')
print("Model, scaler, dan threshold berhasil disimpan.")
print("-" * 50)'''

# --- Untuk Dataset 100k_tampak_depan ---
# Pastikan Anda sudah melatih model dan scaler dengan dataset yang baru
'''joblib.dump(cntr, 'fcm_cluster_centers_100k_depan.joblib')
joblib.dump(scaler, 'fcm_scaler_100k_depan.joblib')
joblib.dump(threshold, 'fcm_threshold_100k_depan.joblib')
print("Model, scaler, dan threshold berhasil disimpan.")
print("-" * 50)'''

# --- Untuk Dataset 100k_tampak_belakang ---
# Pastikan Anda sudah melatih model dan scaler dengan dataset yang baru
'''joblib.dump(cntr, 'fcm_cluster_centers_100k_belakang.joblib')
joblib.dump(scaler, 'fcm_scaler_100k_belakang.joblib')
joblib.dump(threshold, 'fcm_threshold_100k_belakang.joblib')
print("Model, scaler, dan threshold berhasil disimpan.")
print("-" * 50)'''

# --- 5. Fungsi untuk Prediksi ---
def predict_anomaly_fcm(X_data, cluster_centers, threshold_value):
    jarak_ke_pusat_prediksi = np.zeros((X_data.shape[0], cluster_centers.shape[0]))
    for i in range(X_data.shape[0]):
        for j in range(cluster_centers.shape[0]):
            jarak_ke_pusat_prediksi[i, j] = np.linalg.norm(X_data[i] - cluster_centers[j])
            
    jarak_minimal_prediksi = np.min(jarak_ke_pusat_prediksi, axis=1)
    # 1 jika normal (jarak < threshold), -1 jika anomali (jarak >= threshold)
    return np.where(jarak_minimal_prediksi < threshold_value, 1, -1)


# --- 6. Pengujian dan Analisis Hasil ---
prediksi_palsu = predict_anomaly_fcm(X_palsu_scaled, cntr, threshold)
jumlah_palsu = len(prediksi_palsu)
palsu_terdeteksi_anomali = np.sum(prediksi_palsu == -1)
tingkat_deteksi_anomali_palsu = (palsu_terdeteksi_anomali / jumlah_palsu) * 100

prediksi_asli = predict_anomaly_fcm(X_test_asli_scaled, cntr, threshold)
jumlah_asli_test = len(prediksi_asli)
asli_terdeteksi_anomali = np.sum(prediksi_asli == -1)
tingkat_kesalahan_asli = (asli_terdeteksi_anomali / jumlah_asli_test) * 100

print(f"Hasil Pengujian pada Uang Palsu:")
print(f"  - Total sampel: {jumlah_palsu}")
print(f"  - Terdeteksi sebagai anomali (data tidak dikenal): {palsu_terdeteksi_anomali}")
print(f"  - Tingkat deteksi anomali: {tingkat_deteksi_anomali_palsu:.2f}%")
print("-" * 50)

print(f"Hasil Pengujian pada Uang Asli:")
print(f"  - Total sampel: {jumlah_asli_test}")
print(f"  - Salah terdeteksi sebagai anomali (data tidak dikenal): {asli_terdeteksi_anomali}")
print(f"  - Tingkat kesalahan (False Positive): {tingkat_kesalahan_asli:.2f}%")
print("-" * 50)