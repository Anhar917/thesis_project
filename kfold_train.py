import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Import modul K-Fold
from sklearn.model_selection import KFold
from anfis import anfis, membership
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

# --- 1. Memuat Dataset dan Ekstraksi Fitur ---
print("1. Memuat dan Memproses Dataset...")

# Pastikan nama file dan jalurnya sudah benar
X_asli = np.load('/home/pi/MyEnv/anfis_extracted_features/anfis_features_50k_belakang.npy')
Y_target = np.load('/home/pi/MyEnv/anfis_extracted_features/anfis_labels_50k_belakang.npy')

print(f"Dataset asli dimuat: {X_asli.shape[0]} sampel dengan {X_asli.shape[1]} fitur.")

# --- 2. Reduksi Dimensi dengan PCA ---
print("2. Menerapkan PCA untuk Reduksi Dimensi...")
jumlah_fitur_pca = 10

pca = PCA(n_components=jumlah_fitur_pca)
X_pca = pca.fit_transform(X_asli)

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
fold_no = 1

# Loop melalui setiap fold
for train_index, test_index in k_folds.split(X_pca, Y_target): 
    print(f"\n--- Memulai Fold {fold_no} ---")

    # Memisahkan data untuk fold saat ini
    X_train_fold, X_test_fold = X_pca[train_index, :3], X_pca[test_index, :3]
    Y_train_fold, Y_test_fold = Y_target[train_index], Y_target[test_index]
    print(f"Ukuran data pelatihan: {X_train_fold.shape[0]} sampel")
    print(f"Ukuran data pengujian: {X_test_fold.shape[0]} sampel")
    # Membuat model ANFIS yang baru untuk setiap fold
    mf = [
        # Fungsi keanggotaan seperti sebelumnya 
        [
            ['gaussmf', {'mean': -4000., 'sigma': 1500.}],
            ['gaussmf', {'mean': 0., 'sigma': 1500.}],
            ['gaussmf', {'mean': 2000., 'sigma': 1000.}]
            ],
            [
                ['gaussmf', {'mean': -800., 'sigma': 500.}],
                ['gaussmf', {'mean': 0., 'sigma': 500.}],
                ['gaussmf', {'mean': 1500., 'sigma': 800.}]
                ],
                [
                    ['gaussmf', {'mean': -300., 'sigma': 200.}],
                    ['gaussmf', {'mean': 0., 'sigma': 200.}],
                    ['gaussmf', {'mean': 500., 'sigma': 300.}]
                    ]
                    ]

    mf_object = membership.membershipfunction.MemFuncs(mf)
    anfis_model = anfis.ANFIS(X_train_fold, Y_train_fold, mf_object)

    # Latih model
    # Tetapkan tolerance ke nilai yang sangat kecil untuk menghindari bug
    anfis_model.trainHybridJangOffLine(epochs=50, tolerance=1e-15) 

    print(f"Pelatihan selesai untuk Fold {fold_no}.")

    # Prediksi pada data pengujian
    Y_pred_fold = anfis.predict(anfis_model, X_test_fold)
    # Bulatkan prediksi untuk klasifikasi biner
    Y_pred_fold_round = np.round(Y_pred_fold.flatten())
    Y_test_fold_flatten = Y_test_fold.flatten()

    # Menghitung akurasi
    # Jika prediksi di atas 0.5 dianggap 1, selain itu 0
    correct_predictions = np.sum(Y_pred_fold_round == Y_test_fold_flatten)
    total_samples = len(Y_test_fold_flatten)
    accuracy = correct_predictions / total_samples
    # Untuk fokus pada "keaslian", asumsikan uang asli (label 1) adalah kelas positif
    precision_fold = precision_score(Y_test_fold_flatten, Y_pred_fold_round, pos_label=1)
    recall_fold = recall_score(Y_test_fold_flatten, Y_pred_fold_round, pos_label=1)

    print(f"Akurasi untuk Fold {fold_no}: {accuracy * 100:.2f}%")
    print(f"Presisi untuk Fold {fold_no}: {precision_fold * 100:.2f}%")
    print(f"Recall untuk Fold {fold_no}: {recall_fold * 100:.2f}%")
    accuracy_scores.append(accuracy)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    accuracy_scores.append(accuracy)

    fold_no += 1

# --- 4. Menampilkan Hasil Akhir ---
print("\n--- Hasil Akhir Validasi Silang ---")
print(f"Akurasi Rata-rata: {np.mean(accuracy_scores) * 100:.2f}%")
print(f"Standar Deviasi Akurasi: {np.std(accuracy_scores) * 100:.2f}%")
print("-----------------------------------")
print(f"Presisi Rata-rata: {np.mean(precision_scores) * 100:.2f}%")
print(f"Standar Deviasi Presisi: {np.std(precision_scores) * 100:.2f}%")
print("-----------------------------------")
print(f"Recall Rata-rata: {np.mean(recall_scores) * 100:.2f}%")
print(f"Standar Deviasi Recall: {np.std(recall_scores) * 100:.2f}%")