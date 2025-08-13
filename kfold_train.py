import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.decomposition import PCA
# Import modul K-Fold
from sklearn.model_selection import KFold
from anfis import anfis, membership
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

# --- 1. Memuat Dataset dan Ekstraksi Fitur ---
print("1. Memuat dan Memproses Dataset...")

# Pastikan nama file dan jalurnya sudah benar
X_asli = np.load('anfis_extracted_features/anfis_features_50k_depan.npy')
Y_target = np.load('anfis_extracted_features/anfis_labels_50k_depan.npy')

print(f"Dataset asli dimuat: {X_asli.shape[0]} sampel dengan {X_asli.shape[1]} fitur.")

# --- 2. Reduksi Dimensi dengan PCA ---
print("2. Menerapkan PCA untuk Reduksi Dimensi...")
jumlah_fitur_pca = 4

pca = PCA(n_components=jumlah_fitur_pca)
X_pca = pca.fit_transform(X_asli)

print("Mean fitur 1:", np.mean(X_pca[:, 0]))
print("Std fitur 1:", np.std(X_pca[:, 0]))

print("Mean fitur 2:", np.mean(X_pca[:, 1]))
print("Std fitur 2:", np.std(X_pca[:, 1]))

print("Mean fitur 3:", np.mean(X_pca[:, 2]))
print("Std fitur 3:", np.std(X_pca[:, 2]))

print("Mean fitur 4:", np.mean(X_pca[:, 3]))
print("Std fitur 4:", np.std(X_pca[:, 3]))

#print("Mean fitur 5:", np.mean(X_pca[:, 4]))
#print("Std fitur 5:", np.std(X_pca[:, 4]))

'''print("Mean fitur 6:", np.mean(X_pca[:, 5]))
print("Std fitur 6:", np.std(X_pca[:, 5]))'''

print(f"Data berhasil direduksi menjadi: {X_pca.shape[0]} sampel dengan {X_pca.shape[1]} fitur.")

# --- 3. Melakukan Validasi Silang K-Fold ---
print("\n3. Melakukan Validasi Silang K-Fold (5-Fold)...")

# Inisialisasi K-Fold
# Kita akan menggunakan 5 fold, yang berarti 5 kali pelatihan dan pengujian
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Siapkan list untuk menyimpan hasil dari setiap fold
accuracy_scores = []
precision_scores = []
recall_scores = []
all_Y_test = []
all_Y_pred = []
fold_no = 1
# Siapkan plot untuk error
plt.figure(figsize=(15, 8))

# Loop melalui setiap fold
for train_index, test_index in k_folds.split(X_pca, Y_target): 
    print(f"\n--- Memulai Fold {fold_no} ---")

    # Memisahkan data untuk fold saat ini
    X_train_fold, X_test_fold = X_pca[train_index], X_pca[test_index]
    Y_train_fold, Y_test_fold = Y_target[train_index], Y_target[test_index]
    print(f"Ukuran data pelatihan: {X_train_fold.shape[0]} sampel")
    print(f"Ukuran data pengujian: {X_test_fold.shape[0]} sampel")

    # Membuat model ANFIS yang baru untuk setiap fold
    '''mf = [
    # Fungsi keanggotaan untuk Fitur 1
    [
        ['gaussmf', {'mean': -2323., 'sigma': 2323.}],
        ['gaussmf', {'mean': 0., 'sigma': 2323.}],
        ['gaussmf', {'mean': 2323., 'sigma': 2323.}]
    ],
    # Fungsi keanggotaan untuk Fitur 2
    [
        ['gaussmf', {'mean': -869., 'sigma': 869.}],
        ['gaussmf', {'mean': 0., 'sigma': 869.}],
        ['gaussmf', {'mean': 869., 'sigma': 869.}]
    ],
    # Fungsi keanggotaan untuk Fitur 3
    [
        ['gaussmf', {'mean': -377., 'sigma': 377.}],
        ['gaussmf', {'mean': 0., 'sigma': 377.}],
        ['gaussmf', {'mean': 377., 'sigma': 377.}]
    ],
    # Fungsi keanggotaan untuk Fitur 4
    [
        ['gaussmf', {'mean': -159., 'sigma': 159.}],
        ['gaussmf', {'mean': 0., 'sigma': 159.}],
        ['gaussmf', {'mean': 159., 'sigma': 159.}]
    ]]'''
    # Fungsi keanggotaan untuk Fitur 5
    '''[
        ['gaussmf', {'mean': -145., 'sigma': 145.}],
        ['gaussmf', {'mean': 0., 'sigma': 145.}],
        ['gaussmf', {'mean': 145., 'sigma': 145.}]
    ]]'''
    # Fungsi keanggotaan untuk Fitur 6
    '''[
        ['gaussmf', {'mean': -92., 'sigma': 92.}],
        ['gaussmf', {'mean': 0., 'sigma': 92.}],
        ['gaussmf', {'mean': 92., 'sigma': 92.}]
    ]'''
    
    # mf dengan hyperparameter tuning untuk meningkatkan sensitivitas terhadap ciri uang asli
    mf = [
    # Fungsi keanggotaan untuk Fitur 1
    [
        ['gaussmf', {'mean': -5894., 'sigma': 8841.}], # sigma dilebarkan (5894 * 1.5)
        ['gaussmf', {'mean': 0., 'sigma': 2947.}],     # sigma disempitkan (5894 / 2)
        ['gaussmf', {'mean': 5894., 'sigma': 8841.}]
    ],
    # Fungsi keanggotaan untuk Fitur 2
    [
        ['gaussmf', {'mean': -647., 'sigma': 971.}],
        ['gaussmf', {'mean': 0., 'sigma': 323.5}],
        ['gaussmf', {'mean': 647., 'sigma': 971.}]
    ],
    # Fungsi keanggotaan untuk Fitur 3
    [
        ['gaussmf', {'mean': -12.5, 'sigma': 18.75}],
        ['gaussmf', {'mean': 0., 'sigma': 6.25}],
        ['gaussmf', {'mean': 12.5, 'sigma': 18.75}]
    ],
    # Fungsi keanggotaan untuk Fitur 4
    [
        ['gaussmf', {'mean': -8.1, 'sigma': 12.15}],
        ['gaussmf', {'mean': 0., 'sigma': 4.05}],
        ['gaussmf', {'mean': 8.1, 'sigma': 12.15}]
    ]]
    # Fungsi keanggotaan untuk Fitur 5
    '''[
        ['gaussmf', {'mean': -7.6, 'sigma': 11.4}],
        ['gaussmf', {'mean': 0., 'sigma': 3.8}],
        ['gaussmf', {'mean': 7.6, 'sigma': 11.4}]
    ]]'''
    

    mf_object = membership.membershipfunction.MemFuncs(mf)
    anfis_model = anfis.ANFIS(X_train_fold, Y_train_fold, mf_object)

    # Latih model
    # Tetapkan tolerance ke nilai yang sangat kecil untuk menghindari bug
    anfis_model.trainHybridJangOffLine(epochs=10, tolerance=1e-15) 
    # Plot error pelatihan untuk fold ini
    plt.plot(anfis_model.errors, label=f'Fold {fold_no}')

    print(f"Pelatihan selesai untuk Fold {fold_no}.")

    # Prediksi pada data pengujian
    Y_pred_fold = anfis.predict(anfis_model, X_test_fold)
    # Bulatkan prediksi untuk klasifikasi biner
    Y_pred_fold_round = np.round(Y_pred_fold.flatten())
    Y_test_fold_flatten = Y_test_fold.flatten()
    print("Nilai unik di Y_test_fold_flatten:", np.unique(Y_test_fold_flatten))
    # Tambahkan data dari fold saat ini ke list keseluruhan
    all_Y_test.extend(Y_test_fold_flatten)
    all_Y_pred.extend(Y_pred_fold_round)

    # Menghitung akurasi
    # Jika prediksi di atas 0.5 dianggap 1, selain itu 0
    correct_predictions = np.sum(Y_pred_fold_round == Y_test_fold_flatten)
    total_samples = len(Y_test_fold_flatten)
    accuracy = correct_predictions / total_samples
    # Untuk fokus pada "keaslian", asumsikan uang asli (label 1) adalah kelas positif
    #precision_fold = precision_score(Y_test_fold_flatten, Y_pred_fold_round, pos_label=1)
    #recall_fold = recall_score(Y_test_fold_flatten, Y_pred_fold_round, pos_label=1)
    precision_fold, recall_fold, fscore_fold, _ = precision_recall_fscore_support(Y_test_fold_flatten, Y_pred_fold_round, average='macro', zero_division=0)

    print(f"Akurasi untuk Fold {fold_no}: {accuracy * 100:.2f}%")
    print(f"Presisi untuk Fold {fold_no}: {precision_fold * 100:.2f}%")
    print(f"Recall untuk Fold {fold_no}: {recall_fold * 100:.2f}%")
    accuracy_scores.append(accuracy)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)

    fold_no += 1

# --- Plot Error Pelatihan per Epoch untuk Semua Fold ---
plt.title('Error Pelatihan ANFIS per Epoch (Semua Folds)')
plt.xlabel('Epoch')
plt.ylabel('Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('error_pelatihan_keseluruhan.png')
plt.close()

# --- Plot Prediksi vs. Data Asli untuk Semua Fold ---
plt.figure(figsize=(15, 8))
plt.scatter(all_Y_test, all_Y_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Prediksi Model vs. Data Asli (Keseluruhan Folds)')
plt.xlabel('Data Asli (Y_test)')
plt.ylabel('Prediksi Model (Y_pred)')
plt.grid(True)
plt.savefig('prediksi_vs_asli_keseluruhan.png')
plt.close()

# --- 4. Menampilkan Hasil Akhir ---
print("\n--- Hasil Akhir Validasi Silang untuk Uang 50 Ribu Tampak Depan ---")
print(f"Akurasi Rata-rata: {np.mean(accuracy_scores) * 100:.2f}%")
print(f"Standar Deviasi Akurasi: {np.std(accuracy_scores) * 100:.2f}%")
print("-----------------------------------")
print(f"Presisi Rata-rata: {np.mean(precision_scores) * 100:.2f}%")
print(f"Standar Deviasi Presisi: {np.std(precision_scores) * 100:.2f}%")
print("-----------------------------------")
print(f"Recall Rata-rata: {np.mean(recall_scores) * 100:.2f}%")
print(f"Standar Deviasi Recall: {np.std(recall_scores) * 100:.2f}%")

# Nama file untuk menyimpan model
#nama_file_model = 'anfis_model_50k_belakang.pkl'
nama_file_model = 'anfis_model_50k_depan.pkl'

# Tentukan path lengkap untuk menyimpan file
path_simpan = os.path.join(os.getcwd(), 'trained_models', nama_file_model)
# Pastikan direktori 'trained_models' ada
os.makedirs(os.path.dirname(path_simpan), exist_ok=True)
# Simpan model ANFIS ke dalam file
with open(path_simpan, 'wb') as file:
    pickle.dump(anfis_model, file)

print(f"\nModel ANFIS berhasil disimpan di: {path_simpan}")