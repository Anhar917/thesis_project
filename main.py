import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
import re
import time
from skimage.feature import graycomatrix, graycoprops

def ambil_gambar_dari_webcam(base_dataset_dir='Dataset Uang Kertas'):
    
    print("\n--- Mode Pengambilan Gambar Dataset dari Webcam ---")

    # 1. Pilihan Asli/Palsu (Ini akan menjadi subfolder 'Asli' atau 'Palsu')
    while True:
        label_input = input("Masukkan label (asli/palsu): ").lower().strip()
        if label_input in ['asli', 'palsu']:
            label_folder = "Asli" if label_input == "asli" else "Palsu"
            break
        else:
            print("Input tidak valid. Harap masukkan 'asli' atau 'palsu'.")
    # 2. Pilihan Pecahan Uang (Ini akan menjadi subfolder '50K' atau '100K')
    while True:
        pecahan_input = input("Masukkan pecahan uang (50K/100K): ").lower().strip()
        if pecahan_input in ['50k', '100k']:
            pecahan_folder = pecahan_input.upper() # Jadi '50K' atau '100K'
            break
        else:
            print("Input tidak valid. Harap masukkan '50k' atau '100k'.")
    # 3. Pilihan Sisi Uang (Ini akan mempengaruhi penamaan file dan subfolder 'Tampak Depan'/'Tampak Belakang')
    while True:
        sisi_input = input("Masukkan sisi uang (depan/belakang): ").lower().strip()
        if sisi_input in ['depan', 'belakang']:
            sisi_folder = f"{sisi_input.capitalize()}" # 'Tampak Depan' atau 'Tampak Belakang'
            break
        else:
            print("Input tidak valid. Harap masukkan 'depan' atau 'belakang'.")
    # Path lengkap ke folder penyimpanan
    # Contoh: Dataset Uang Kertas/Asli/50K/Tampak Depan
    save_dir = os.path.join(base_dataset_dir, label_folder, pecahan_folder, sisi_folder)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Gambar akan disimpan di: {save_dir}")

    # Logika penomoran otomatis yang sudah Anda gunakan
    # Cari file yang formatnya seperti "uang 50K_tampak belakang(1).jpg" atau "uang_100K_tampak_depan(5).jpg"

    base_file_prefix = f"uang {pecahan_folder.lower()}_tampak {sisi_input}"
    # Regex untuk mencari angka di akhir nama file dengan format (N).jpg
    # pattern = re.compile(r'\((?P<number>\d+)\)\.jpg$', re.IGNORECASE) # Untuk format (N).jpg
    pattern = re.compile(rf'^{re.escape(base_file_prefix)}\((\d+)\)\.jpg$', re.IGNORECASE)
    max_number = 0
    all_files_in_dir = os.listdir(save_dir)

    for file_name in all_files_in_dir:
        match = pattern.match(file_name)
        if match:
            try:
                num = int(match.group(1)) # Ambil angka dari grup tangkapan pertama
                if num > max_number:
                    max_number = num
            except ValueError:
                continue # Abaikan file yang tidak sesuai format penomoran
    next_capture_number = max_number + 1

    gambar_tertangkap = None
    file_path_tersimpan = None

    cap = cv2.VideoCapture(0) # Buka webcam default (0)

    # Coba set resolusi yang diinginkan
    desired_width = 854
    desired_height = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Verifikasi resolusi aktual yang diterapkan oleh kamera
    actual_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam. Pastikan webcam terhubung dan driver terinstal.")
        return None # Mengembalikan None jika webcam tidak bisa dibuka, penting untuk penanganan di main
    else:
        
        print("\nTekan 's' untuk menyimpan gambar.")
        print("Tekan 'q' untuk keluar.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari webcam.")
                break

            # Tampilkan informasi di preview (opsional, untuk user guidance)
            display_text = f"Simpan ke: {label_folder}/{pecahan_folder}/{sisi_folder} (s: simpan {next_capture_number}, q: keluar)"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Live Webcam Preview', frame)
            cv2.resizeWindow('Live Camera Feed', actual_cam_width, actual_cam_height)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Simpan gambar dengan format yang diminta
                # Contoh: "uang_50k_tampak_depan(1).jpg"
                file_name = f"{base_file_prefix}({next_capture_number}).jpg"
                file_path = os.path.join(save_dir, file_name)
                cv2.imwrite(file_path, frame)
                print(f"Gambar disimpan: {file_path}")
                # KEMBALIKAN PATH FILE YANG BARU SAJA DISIMPAN
                cap.release()
                cv2.destroyAllWindows()
                return file_path # Mengembalikan path file yang berhasil disimpan
            elif key == ord('q'):
                print("Keluar dari mode pengambilan gambar.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return None

def identifikasi_pecahan_berdasarkan_cr_histogram(cr_channel_input):
    
    # --- Pengecekan Input ---
    if cr_channel_input is None or not isinstance(cr_channel_input, np.ndarray) or cr_channel_input.size == 0:
        print("ERROR: Input cr_channel_input tidak valid (None, bukan array, atau kosong).")
        return 'unknown'
    
    # Pastikan dtype adalah numerik (misalnya uint8 atau float).
    if cr_channel_input.dtype == object:
        print("ERROR: cr_channel_input memiliki dtype 'object'. Tidak dapat memproses histogram.")
        return 'unknown'

    # --- Filtering Piksel Uang ---
    # Saring piksel yang tidak nol (bagian uang yang sebenarnya setelah masking).
    # Ini untuk menghindari noise dari piksel hitam (0) di background setelah masking.
    pixels_in_mask = cr_channel_input[cr_channel_input > 0] 
    
    if pixels_in_mask.size == 0:
        print("Peringatan: Tidak ada piksel uang yang valid di cr_channel_input setelah filtering. Mungkin masking terlalu agresif.")
        return 'unknown'

    # --- Logika Identifikasi Berdasarkan Persentase Piksel Cr ---

    # Total jumlah piksel yang valid di area uang (setelah masking)
    total_valid_pixels = pixels_in_mask.size

    # Ambang batas nilai Cr untuk 100 ribu (misalnya, piksel yang lebih terang di saluran Cr)
    cr_threshold_100k = 200 
    
    # Hitung piksel yang nilainya di atas ambang batas Cr_threshold_100k
    pixels_above_threshold = pixels_in_mask[pixels_in_mask > cr_threshold_100k]
    
    # Hitung persentase piksel yang nilainya di atas ambang batas
    if total_valid_pixels > 0:
        percentage_above_threshold = (pixels_above_threshold.size / total_valid_pixels) * 100
    else:
        percentage_above_threshold = 0 # Hindari ZeroDivisionError

    #print(f"DEBUG: Persentase piksel Cr > {cr_threshold_100k}: {percentage_above_threshold:.2f}%")

    # Batas persentase untuk mengklasifikasikan sebagai 100 ribu
    percentage_threshold_for_100k = 1.0 

    if percentage_above_threshold >= percentage_threshold_for_100k:
        return '100K'
    else:
        return '50K'


def buat_mask_uang_kertas(gambar_bgr):
    """
    Membuat mask biner yang memisahkan uang kertas dari background.
    """
    gambar_bgr_internal = cv2.cvtColor(gambar_bgr, cv2.COLOR_RGB2BGR)
    gambar_gray = cv2.cvtColor(gambar_bgr_internal, cv2.COLOR_BGR2GRAY)
    # --- Metode 1: Global Thresholding pada Grayscale (Paling Sederhana) ---
    
    threshold_value = 49 
    _, mask = cv2.threshold(gambar_gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    #print(f"Mask dibuat dengan Global Threshold (GRAY) = {threshold_value}.")
    # --- Operasi Morfologi untuk Membersihkan Mask ---
    # Ini sangat penting untuk mendapatkan mask yang rapi.
    kernel = np.ones((5,5),np.uint8) # Kernel persegi 5x5
    # MORPH_OPEN: Menghilangkan noise kecil (titik-titik putih di area hitam)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    # MORPH_CLOSE: Menutup lubang kecil di dalam objek (titik-titik hitam di area putih)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    #print("Mask dibersihkan dengan operasi morfologi (OPEN dan CLOSE).")

    return mask


def hitung_fitur_glcm(image_segment_cr):
    
    if image_segment_cr is None or image_segment_cr.size == 0 or np.max(image_segment_cr) == np.min(image_segment_cr):
        print("DEBUG GLCM: Segmen kosong atau seragam.")
        return {
            'glcm_contrast': 0.0, 'glcm_correlation': 0.0,
            'glcm_energy': 0.0, 'glcm_homogeneity': 0.0
        }

    img_min = np.min(image_segment_cr[image_segment_cr > 0]) if np.any(image_segment_cr > 0) else 0
    img_max = np.max(image_segment_cr)
    
    if img_max == img_min:
        normalized_segment = image_segment_cr.astype(np.uint8)
    else:
        normalized_segment = ((image_segment_cr - img_min) * (255 / (img_max - img_min))).astype(np.uint8)

    # Menghitung GLCM
    # distances: Jarak antar piksel. [1] berarti hanya mempertimbangkan tetangga langsung.
    # angles: Arah piksel. [0, np.pi/4, np.pi/2, 3*np.pi/4] untuk 0, 45, 90, 135 derajat.
    # levels: Jumlah level keabuan. Karena cr_channel_masked dari 0-255, pakai 256.
    # symmetric: True untuk membuat matriks simetris (mempertimbangkan kedua arah)
    # normed: True untuk normalisasi (penting agar fitur konsisten antar gambar)
    
    try:
        glcm = graycomatrix(normalized_segment, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
                            symmetric=True, normed=True)
        # MENGAMBIL RATA-RATA DARI 4 ARAH UNTUK SETIAP FITUR
        # graycoprops akan mengembalikan array (1, 4). Kita ambil rata-rata dari kolom-kolomnya.
        
        # Contoh: graycoprops(glcm, 'contrast') akan menghasilkan array seperti [[c0, c45, c90, c135]]
        # np.mean(..., axis=1) akan merata-ratakan nilai-nilai ini.
        contrast = np.mean(graycoprops(glcm, 'contrast'))[0] # Mengambil rata-rata dari 4 arah, lalu ambil elemen pertamanya
        correlation = np.mean(graycoprops(glcm, 'correlation'))[0]
        energy = np.mean(graycoprops(glcm, 'energy'))[0]
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))[0]

        return {
            'glcm_contrast': contrast,
            'glcm_correlation': correlation,
            'glcm_energy': energy,
            'glcm_homogeneity': homogeneity
        }
    except Exception as e: # Gunakan Exception yang lebih umum
        print(f"!!! KRITIS !!! ERROR TERTANGKAP DI HITUNG_FITUR_GLCM (DI DALAM TRY): {e}")
        return {
            'glcm_contrast': 0.0,
            'glcm_correlation': 0.0,
            'glcm_energy': 0.0,
            'glcm_homogeneity': 0.0
        }


def ekstraksi_bentuk_dan_pemberian_label_50K_Depan(final_segmentation_cleaned, cr_channel_masked):
    
    min_area_threshold=200
    max_area_threshold=30000
    final_segmentation_cleaned = final_segmentation_cleaned.astype(np.uint8) * 255
    # Temukan kontur dari objek-objek putih
    # RETR_EXTERNAL: Hanya mengambil kontur terluar (bagus untuk menghindari lubang internal sebagai kontur terpisah)
    contours, _ = cv2.findContours(final_segmentation_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_objects_50k_Depan = {}
    #print(f"Ditemukan {len(contours)} kontur.")

    kontur_image = np.zeros_like(final_segmentation_cleaned)
    # Jika Anda ingin melihat kontur di atas gambar segmentasi asli:
    kontur_image = final_segmentation_cleaned.copy()
    # Gambar semua kontur yang ditemukan dengan warna (misalnya hijau)
    # Menggambar dengan tebal 2 atau 3 agar terlihat jelas
    # Kontur akan digambar sebagai garis luar
    cv2.drawContours(kontur_image, contours, -1, (255, 255, 255), 2) # Menggambar di gambar hitam, kontur putih
    '''plt.figure(figsize=(12,5))
    plt.imshow(kontur_image, cmap='gray') # Gunakan cmap='gray' untuk gambar biner
    plt.title('Semua Kontur Ditemukan oleh findContours')
    plt.axis('off')
    plt.show()'''

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Dapatkan bounding box (x, y, lebar, tinggi)
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- VISUALISASI KONTUR SAAT INI DI ATAS GAMBAR SEGMENTASI ---
        # Buat salinan gambar basis untuk setiap iterasi agar hanya kontur saat ini yang disorot
        current_contour_display = cv2.cvtColor(kontur_image, cv2.COLOR_GRAY2BGR) # Konversi ke BGR untuk warna
        # Gambar kontur saat ini dengan warna cerah (misal: hijau)
        cv2.drawContours(current_contour_display, [contour], -1, (0, 255, 0), 2) # Warna hijau, tebal 2
        # Gambar bounding box juga bisa membantu
        cv2.rectangle(current_contour_display, (x, y), (x+w, y+h), (255, 0, 0), 2)

        '''plt.figure(figsize=(12,5))
        plt.imshow(current_contour_display) # imshow akan tahu ini BGR
        plt.title(f'Kontur {i} (Area: {area:.1f}) pada Segmentasi')
        plt.axis('off')
        plt.show()'''

        # Filter kontur berdasarkan area untuk menghilangkan noise dan sisa-sisa kecil
        if area < min_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu kecil: {area} < {min_area_threshold})")
            continue
        elif area > max_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu besar: {area} > {max_area_threshold})")
            continue
        else:
            print(f"Kontur {i} ditemukan (Area: {area})")
            
        
        # Dapatkan ciri bentuk dasar
        aspect_ratio = float(w) / h if h != 0 else 0
        solidity = float(area) / (w * h) if (w*h) != 0 else 0
        # Contoh Hu Moments (untuk invarian rotasi, skala, translasi)
        # Hitung momen dulu
        M = cv2.moments(contour)
        if M["m00"] == 0: # Hindari pembagian nol jika momen area 0 (sangat kecil atau garis)
            print(f"Kontur {i} diabaikan (momen area nol).")
            continue
        hu_moments = cv2.HuMoments(M).flatten()
        # Hu moments seringkali diambil log untuk skala yang lebih baik
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        # Buat mask untuk objek individu
        object_mask = np.zeros_like(final_segmentation_cleaned)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

        # --- LOGIKA IDENTIFIKASI/PEMBERIAN LABEL ---
        center_x_norm = (x + w/2) / final_segmentation_cleaned.shape[1]
        center_y_norm = (y + h/2) / final_segmentation_cleaned.shape[0]
        '''print(f"\tAspectRatio={aspect_ratio:.2f}, \n\tSolidity={solidity:.2f}")
        print(f"\tCenter Norm: ({center_x_norm:.2f}, {center_y_norm:.2f})")
        print(f"\tHu Moments (log): {hu_moments_log.tolist()}")'''

        label = "Lain-lain/Noise"
        if 0.1 < center_x_norm < 0.8 and center_y_norm > 0.6 and 600 < area < 5000 and 1.0 < aspect_ratio < 6.0:
            label = "Pola Patokan Kecil"
        elif 0.2 < center_x_norm < 0.8  and center_y_norm < 0.7 and 5000 < area < 50000 and 0.5 < aspect_ratio < 5.0:
            label = "Pola Patokan Besar"
        #print(f"\tLabel yang ditentukan: {label}")

        # Tambahkan ke dictionary jika itu salah satu ciri utama
        if label in ["Pola Patokan Kecil", 'Pola Patokan Besar'] and label not in labeled_objects_50k_Depan:
            labeled_objects_50k_Depan[label] = {
                'mask': object_mask,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'hu_moments': hu_moments_log.tolist(),
                'center_x_norm': center_x_norm,
                'center_y_norm': center_y_norm }
            #print(f"Objek '{label}' berhasil ditambahkan.")
    
    final_features_for_anfis = {} # Dictionary untuk menyimpan semua fitur yang akan jadi input ANFIS 
    ciri_names_50K_Depan = ['Pola Patokan Kecil', 'Pola Patokan Besar']
    for ciri_name in ciri_names_50K_Depan:
        if ciri_name in labeled_objects_50k_Depan: # Pastikan ciri tersebut berhasil ditemukan dan dilabeli
            data_ciri = labeled_objects_50k_Depan[ciri_name]
            mask_ciri = data_ciri['mask']
            # --- MULAI BAGIAN BARU: EKSTRAKSI FITUR BENTUK DAN TEKSTUR ---
            
            # 1. Simpan Fitur Bentuk yang Sudah Dihitung
            # (Anda sudah memiliki ini dari langkah pelabelan sebelumnya)
            final_features_for_anfis[f'{ciri_name}_area'] = data_ciri['area']
            #print(f"FINAL FEATURES FOR ANFIS")
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = data_ciri['aspect_ratio']
            final_features_for_anfis[f'{ciri_name}_solidity'] = data_ciri['solidity']
            for i, hu in enumerate(data_ciri['hu_moments']):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = hu
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = data_ciri['center_x_norm']
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = data_ciri['center_y_norm']

            # 2. Ekstraksi Fitur Tekstur (GLCM)
            ciri_area_cr_masked = cv2.bitwise_and(cr_channel_masked, cr_channel_masked, mask=mask_ciri)
            # PENTING: Debug visual di sini untuk memastikan ciri_area_cr_masked terlihat benar
            '''print(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            print(f"DEBUG INPUT GLCM: Max piksel: {np.max(ciri_area_cr_masked)}")'''
            glcm_features = hitung_fitur_glcm(ciri_area_cr_masked)
            '''plt.figure(figsize=(12,5))
            plt.imshow(ciri_area_cr_masked, cmap='gray')
            plt.title(f'Masked GLCM Area for {ciri_name}')
            plt.show()'''

            print(f"FINAL FEATURES FOR ANFIS GLCM")
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = glcm_features['glcm_contrast']
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = glcm_features['glcm_correlation']
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = glcm_features['glcm_energy']
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = glcm_features['glcm_homogeneity']
            #print(f"Ini adalah: {final_features_for_anfis}")
        else:
            # Jika ciri tidak ditemukan, isi dengan nilai default (misal 0.0)
            # Ini PENTING agar jumlah fitur selalu konsisten untuk ANFIS
            # Buat placeholder untuk setiap fitur spesifik:
            final_features_for_anfis[f'{ciri_name}_area'] = 0.0
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = 0.0
            final_features_for_anfis[f'{ciri_name}_solidity'] = 0.0
            for i in range(7):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = 0.0
    
    ordered_feature_keys_50K = []
    for ciri_name in ciri_names_50K_Depan:
        ordered_feature_keys_50K.extend([
            f'{ciri_name}_area', f'{ciri_name}_aspect_ratio', f'{ciri_name}_solidity',
        ] + [f'{ciri_name}_hu_moment_{i}' for i in range(7)] + [ # 7 Hu Moments
            f'{ciri_name}_center_x_norm', f'{ciri_name}_center_y_norm',
            f'{ciri_name}_glcm_contrast', f'{ciri_name}_glcm_correlation',
            f'{ciri_name}_glcm_energy', f'{ciri_name}_glcm_homogeneity'
        ])
    #print(f"FINAL ANFIS INPUT VECTOR")
    final_anfis_input_vector = [final_features_for_anfis[key] for key in ordered_feature_keys_50K]
    #print(f"Ini adalah: {final_anfis_input_vector}")
    return np.array(final_anfis_input_vector)

def ekstraksi_bentuk_dan_pemberian_label_50K(final_segmentation_cleaned, cr_channel_masked):
    
    min_area_threshold=200
    max_area_threshold=10000
    final_segmentation_cleaned = final_segmentation_cleaned.astype(np.uint8) * 255
    # Temukan kontur dari objek-objek putih
    # RETR_EXTERNAL: Hanya mengambil kontur terluar (bagus untuk menghindari lubang internal sebagai kontur terpisah)
    contours, _ = cv2.findContours(final_segmentation_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_objects_50k = {}
    #print(f"Ditemukan {len(contours)} kontur.") # Print jumlah kontur yang ditemukan
    
    # --- VISUALISASI KONTUR YANG DITEMUKAN ---
    kontur_image = np.zeros_like(final_segmentation_cleaned)
    # Jika Anda ingin melihat kontur di atas gambar segmentasi asli:
    kontur_image = final_segmentation_cleaned.copy()
    # Gambar semua kontur yang ditemukan dengan warna (misalnya hijau)
    # Menggambar dengan tebal 2 atau 3 agar terlihat jelas
    # Kontur akan digambar sebagai garis luar
    cv2.drawContours(kontur_image, contours, -1, (255, 255, 255), 2) # Menggambar di gambar hitam, kontur putih
    '''plt.figure(figsize=(12,5))
    plt.imshow(kontur_image, cmap='gray') # Gunakan cmap='gray' untuk gambar biner
    plt.title('Semua Kontur Ditemukan oleh findContours')
    plt.axis('off')
    plt.show()'''

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Dapatkan bounding box (x, y, lebar, tinggi)
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- VISUALISASI KONTUR SAAT INI DI ATAS GAMBAR SEGMENTASI ---
        # Buat salinan gambar basis untuk setiap iterasi agar hanya kontur saat ini yang disorot
        current_contour_display = cv2.cvtColor(kontur_image, cv2.COLOR_GRAY2BGR) # Konversi ke BGR untuk warna
        # Gambar kontur saat ini dengan warna cerah (misal: hijau)
        cv2.drawContours(current_contour_display, [contour], -1, (0, 255, 0), 2) # Warna hijau, tebal 2
        # Gambar bounding box juga bisa membantu
        cv2.rectangle(current_contour_display, (x, y), (x+w, y+h), (255, 0, 0), 2) # Warna biru, tebal 2
        
        '''plt.figure(figsize=(12,5))
        plt.imshow(current_contour_display) # imshow akan tahu ini BGR
        plt.title(f'Kontur {i} (Area: {area:.1f}) pada Segmentasi')
        plt.axis('off')
        plt.show()''' # Tampilkan setiap kontur satu per satu dengan highlight

        # Filter kontur berdasarkan area untuk menghilangkan noise dan sisa-sisa kecil
        if area < min_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu kecil: {area} < {min_area_threshold})")
            continue
        elif area > max_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu besar: {area} > {max_area_threshold})")
            continue
        else:
            print(f"Kontur {i} ditemukan (Area: {area})")
            
        
        # Dapatkan ciri bentuk dasar
        aspect_ratio = float(w) / h if h != 0 else 0
        solidity = float(area) / (w * h) if (w*h) != 0 else 0
        # Contoh Hu Moments (untuk invarian rotasi, skala, translasi)
        # Hitung momen dulu
        M = cv2.moments(contour)
        if M["m00"] == 0: # Hindari pembagian nol jika momen area 0 (sangat kecil atau garis)
            print(f"Kontur {i} diabaikan (momen area nol).")
            continue
        hu_moments = cv2.HuMoments(M).flatten()
        # Hu moments seringkali diambil log untuk skala yang lebih baik
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        # Buat mask untuk objek individu
        object_mask = np.zeros_like(final_segmentation_cleaned)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

        # --- LOGIKA IDENTIFIKASI/PEMBERIAN LABEL ---
        
        center_x_norm = (x + w/2) / final_segmentation_cleaned.shape[1]
        center_y_norm = (y + h/2) / final_segmentation_cleaned.shape[0]
        '''print(f"\tAspectRatio={aspect_ratio:.2f}, \n\tSolidity={solidity:.2f}")
        print(f"\tCenter Norm: ({center_x_norm:.2f}, {center_y_norm:.2f})")
        print(f"\tHu Moments (log): {hu_moments_log.tolist()}")'''

        label = "Lain-lain/Noise"
        # Logika untuk Angka Nominal (misal, 50000 atau 100000)
        # Biasanya di kanan bawah, area tertentu, rasio aspek tertentu
        if center_x_norm > 0.5 and center_y_norm > 0.6 and 2500 < area < 8000 and 2.0 < aspect_ratio < 6.0:
            label = "Angka Nominal"
        # Logika untuk Logo BI (misal, di kiri tengah, area tertentu, rasio aspek mendekati 1)
        elif center_x_norm < 0.5 and center_y_norm > 0.3 and center_y_norm < 0.7 and 2500 < area < 6000 and 0.25 < aspect_ratio < 1.0:
            label = "Logo BI"
        # Logika untuk Tulisan "BANK INDONESIA" (misal, di kanan atas, area tertentu, rasio aspek sangat tinggi)
        elif 0.75 > center_x_norm > 0.5 and center_y_norm < 0.5 and 200 < area < 550 and aspect_ratio > 2.0:
            label = "Bank"
        elif center_x_norm > 0.6 and center_y_norm < 0.5 and 550 < area < 1400 and aspect_ratio > 2.0:
            label = "Indonesia"
        elif center_x_norm > 0.5 and center_y_norm < 0.5 and 1400 < area < 2000 and aspect_ratio > 4.0:
            label = "Bank Indonesia"
        #print(f"\tLabel yang ditentukan: {label}")
        # Tambahkan ke dictionary jika itu salah satu ciri utama
        if label in ["Logo BI", "Angka Nominal", "Bank", "Indonesia", "Bank Indonesia"] and label not in labeled_objects_50k:
            labeled_objects_50k[label] = {
                'mask': object_mask,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'hu_moments': hu_moments_log.tolist(),
                'center_x_norm': center_x_norm,
                'center_y_norm': center_y_norm } # Simpan sebagai list untuk kemudahan
                
                # Anda bisa tambahkan ciri bentuk lain di sini
            # Debug: Tampilkan objek yang teridentifikasi
            print(f"Objek '{label}' berhasil ditambahkan.")
    final_features_for_anfis = {} # Dictionary untuk menyimpan semua fitur yang akan jadi input ANFIS 
    ciri_names_50K = ['Angka Nominal', 'Logo BI', 'Bank', 'Indonesia', 'Bank Indonesia'] # Tambahkan jika Anda punya label ini
    for ciri_name in ciri_names_50K:
        if ciri_name in labeled_objects_50k: # Pastikan ciri tersebut berhasil ditemukan dan dilabeli
            data_ciri = labeled_objects_50k[ciri_name]
            mask_ciri = data_ciri['mask'] # Ini dia mask untuk ciri spesifik!

            # --- MULAI BAGIAN BARU: EKSTRAKSI FITUR BENTUK DAN TEKSTUR ---
            
            # 1. Simpan Fitur Bentuk yang Sudah Dihitung
            # (Anda sudah memiliki ini dari langkah pelabelan sebelumnya)
            final_features_for_anfis[f'{ciri_name}_area'] = data_ciri['area']
            #print(f"FINAL FEATURES FOR ANFIS")
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = data_ciri['aspect_ratio']
            final_features_for_anfis[f'{ciri_name}_solidity'] = data_ciri['solidity']
            for i, hu in enumerate(data_ciri['hu_moments']):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = hu
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = data_ciri['center_x_norm']
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = data_ciri['center_y_norm']

            # 2. Ekstraksi Fitur Tekstur (GLCM)
            ciri_area_cr_masked = cv2.bitwise_and(cr_channel_masked, cr_channel_masked, mask=mask_ciri)
            # PENTING: Debug visual di sini untuk memastikan ciri_area_cr_masked terlihat benarprint(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            '''print(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            print(f"DEBUG INPUT GLCM: Max piksel: {np.max(ciri_area_cr_masked)}")'''
            glcm_features = hitung_fitur_glcm(ciri_area_cr_masked)
            '''plt.figure(figsize=(12,5))
            plt.imshow(ciri_area_cr_masked, cmap='gray')
            plt.title(f'Masked GLCM Area for {ciri_name}')
            plt.show()'''

            #print(f"FINAL FEATURES FOR ANFIS GLCM")
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = glcm_features['glcm_contrast']
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = glcm_features['glcm_correlation']
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = glcm_features['glcm_energy']
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = glcm_features['glcm_homogeneity']
            #print(f"Ini adalah: {final_features_for_anfis}")
        else:
            # Jika ciri tidak ditemukan, isi dengan nilai default (misal 0.0)
            # Ini PENTING agar jumlah fitur selalu konsisten untuk ANFIS
            # Hitung berapa banyak fitur yang diharapkan untuk satu ciri (misal 5 bentuk + 7 Hu + 2 center + 4 GLCM = 18 fitur per ciri)
            num_features_per_ciri = 5 + 7 + 4 # sesuaikan dengan jumlah fitur yang Anda ekstrak
                
            # Buat placeholder untuk setiap fitur spesifik:
            final_features_for_anfis[f'{ciri_name}_area'] = 0.0
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = 0.0
            final_features_for_anfis[f'{ciri_name}_solidity'] = 0.0
            for i in range(7):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = 0.0

    # Terakhir, ubah dictionary `final_features_for_anfis` menjadi array/list numerik
    # yang urutannya KONSISTEN. Ini sangat penting untuk ANFIS.
    # Anda perlu mendefinisikan urutan kunci fitur Anda.
    # Contoh urutan kunci untuk 50K:'''
    ordered_feature_keys_50K = []
    for ciri_name in ciri_names_50K:
        ordered_feature_keys_50K.extend([
            f'{ciri_name}_area', f'{ciri_name}_aspect_ratio', f'{ciri_name}_solidity',
        ] + [f'{ciri_name}_hu_moment_{i}' for i in range(7)] + [ # 7 Hu Moments
            f'{ciri_name}_center_x_norm', f'{ciri_name}_center_y_norm',
            f'{ciri_name}_glcm_contrast', f'{ciri_name}_glcm_correlation',
            f'{ciri_name}_glcm_energy', f'{ciri_name}_glcm_homogeneity'
        ])
    #print(f"FINAL ANFIS INPUT VECTOR")
    final_anfis_input_vector = [final_features_for_anfis[key] for key in ordered_feature_keys_50K]
    #print(f"Ini adalah: {final_anfis_input_vector}")

    return np.array(final_anfis_input_vector) # ANFIS biasanya butuh numpy array

def ekstraksi_bentuk_dan_pemberian_label_100K_Depan(final_segmentation_cleaned, cr_channel_masked):
    
    min_area_threshold=90
    max_area_threshold=7000
    final_segmentation_cleaned = final_segmentation_cleaned.astype(np.uint8) * 255
    # Temukan kontur dari objek-objek putih
    # RETR_EXTERNAL: Hanya mengambil kontur terluar (bagus untuk menghindari lubang internal sebagai kontur terpisah)
    contours, _ = cv2.findContours(final_segmentation_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_objects_100k_Depan = {}
    #print(f"Ditemukan {len(contours)} kontur.")

    # --- VISUALISASI KONTUR YANG DITEMUKAN ---
    # Buat salinan gambar hasil segmentasi untuk digambar
    # Kita akan menggambar kontur di atasnya untuk visualisasi
    kontur_image = np.zeros_like(final_segmentation_cleaned)
    # Jika Anda ingin melihat kontur di atas gambar segmentasi asli:
    kontur_image = final_segmentation_cleaned.copy()
    
    cv2.drawContours(kontur_image, contours, -1, (255, 255, 255), 2) # Menggambar di gambar hitam, kontur putih
    '''plt.figure(figsize=(12,5))
    plt.imshow(kontur_image, cmap='gray') # Gunakan cmap='gray' untuk gambar biner
    plt.title('Semua Kontur Ditemukan oleh findContours')
    plt.axis('off')
    plt.show()'''

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Dapatkan bounding box (x, y, lebar, tinggi)
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- VISUALISASI KONTUR SAAT INI DI ATAS GAMBAR SEGMENTASI ---
        # Buat salinan gambar basis untuk setiap iterasi agar hanya kontur saat ini yang disorot
        current_contour_display = cv2.cvtColor(kontur_image, cv2.COLOR_GRAY2BGR) # Konversi ke BGR untuk warna
        # Gambar kontur saat ini dengan warna cerah (misal: hijau)
        cv2.drawContours(current_contour_display, [contour], -1, (0, 255, 0), 2) # Warna hijau, tebal 2
        # Gambar bounding box juga bisa membantu
        cv2.rectangle(current_contour_display, (x, y), (x+w, y+h), (255, 0, 0), 2) # Warna biru, tebal 2

        '''plt.figure(figsize=(12,5))
        plt.imshow(current_contour_display) # imshow akan tahu ini BGR
        plt.title(f'Kontur {i} (Area: {area:.1f}) pada Segmentasi')
        plt.axis('off')
        plt.show()'''

        # Filter kontur berdasarkan area untuk menghilangkan noise dan sisa-sisa kecil
        if area < min_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu kecil: {area} < {min_area_threshold})")
            continue
        elif area > max_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu besar: {area} > {max_area_threshold})")
            continue
        else:
            print(f"Kontur {i} ditemukan (Area: {area})")
            
        
        # Dapatkan ciri bentuk dasar
        aspect_ratio = float(w) / h if h != 0 else 0
        solidity = float(area) / (w * h) if (w*h) != 0 else 0
        # Contoh Hu Moments (untuk invarian rotasi, skala, translasi)
        # Hitung momen dulu
        M = cv2.moments(contour)
        if M["m00"] == 0: # Hindari pembagian nol jika momen area 0 (sangat kecil atau garis)
            print(f"Kontur {i} diabaikan (momen area nol).")
            continue
        hu_moments = cv2.HuMoments(M).flatten()
        # Hu moments seringkali diambil log untuk skala yang lebih baik
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        # Buat mask untuk objek individu
        object_mask = np.zeros_like(final_segmentation_cleaned)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

        center_x_norm = (x + w/2) / final_segmentation_cleaned.shape[1]
        center_y_norm = (y + h/2) / final_segmentation_cleaned.shape[0]
        ''' print(f"\tAspectRatio={aspect_ratio:.2f}, \n\tSolidity={solidity:.2f}")
        print(f"\tCenter Norm: ({center_x_norm:.2f}, {center_y_norm:.2f})")
        print(f"\tHu Moments (log): {hu_moments_log.tolist()}")'''

        label = "Lain-lain/Noise"
        if 0.1 < center_x_norm < 0.2 and center_y_norm > 0.65 and 90 < area < 600 and 0.5 < aspect_ratio < 5.0:
            label = "Pola Patokan 1"
        elif 0.2 < center_x_norm < 0.3 and center_y_norm > 0.65 and 90 < area < 600 and 0.5 < aspect_ratio < 5.0:
            label = "Pola Patokan 2"
        elif 0.3 < center_x_norm < 0.4 and center_y_norm > 0.65 and 90 < area < 600 and 0.5 < aspect_ratio < 5.0:
            label = "Pola Patokan 3"
        #print(f"\tLabel yang ditentukan: {label}")

        # Tambahkan ke dictionary jika itu salah satu ciri utama
        if label in ["Pola Patokan 1", 'Pola Patokan 2', 'Pola Patokan 3'] and label not in labeled_objects_100k_Depan:
            labeled_objects_100k_Depan[label] = {
                'mask': object_mask,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'hu_moments': hu_moments_log.tolist(),
                'center_x_norm': center_x_norm,
                'center_y_norm': center_y_norm }
            print(f"Objek '{label}' berhasil ditambahkan.")
    final_features_for_anfis = {} # Dictionary untuk menyimpan semua fitur yang akan jadi input ANFIS 
    ciri_names_100K_Depan = ['Pola Patokan 1', 'Pola Patokan 2', 'Pola Patokan 3']
    for ciri_name in ciri_names_100K_Depan:
        if ciri_name in labeled_objects_100k_Depan: # Pastikan ciri tersebut berhasil ditemukan dan dilabeli
            data_ciri = labeled_objects_100k_Depan[ciri_name]
            mask_ciri = data_ciri['mask']
            # --- MULAI BAGIAN BARU: EKSTRAKSI FITUR BENTUK DAN TEKSTUR ---
            
            # 1. Simpan Fitur Bentuk yang Sudah Dihitung
            # (Anda sudah memiliki ini dari langkah pelabelan sebelumnya)
            final_features_for_anfis[f'{ciri_name}_area'] = data_ciri['area']
            #print(f"FINAL FEATURES FOR ANFIS")
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = data_ciri['aspect_ratio']
            final_features_for_anfis[f'{ciri_name}_solidity'] = data_ciri['solidity']
            for i, hu in enumerate(data_ciri['hu_moments']):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = hu
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = data_ciri['center_x_norm']
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = data_ciri['center_y_norm']

            # 2. Ekstraksi Fitur Tekstur (GLCM)
            ciri_area_cr_masked = cv2.bitwise_and(cr_channel_masked, cr_channel_masked, mask=mask_ciri)
            # PENTING: Debug visual di sini untuk memastikan ciri_area_cr_masked terlihat benar
            '''print(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            print(f"DEBUG INPUT GLCM: Max piksel: {np.max(ciri_area_cr_masked)}")'''
            glcm_features = hitung_fitur_glcm(ciri_area_cr_masked)
            '''plt.figure(figsize=(12,5))
            plt.imshow(ciri_area_cr_masked, cmap='gray')
            plt.title(f'Masked GLCM Area for {ciri_name}')
            plt.show()'''
            #print(f"FINAL FEATURES FOR ANFIS GLCM")
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = glcm_features['glcm_contrast']
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = glcm_features['glcm_correlation']
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = glcm_features['glcm_energy']
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = glcm_features['glcm_homogeneity']
            #print(f"Ini adalah: {final_features_for_anfis}")
        else:
            # Jika ciri tidak ditemukan, isi dengan nilai default (misal 0.0)
            # Ini PENTING agar jumlah fitur selalu konsisten untuk ANFIS
            # Buat placeholder untuk setiap fitur spesifik:
            final_features_for_anfis[f'{ciri_name}_area'] = 0.0
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = 0.0
            final_features_for_anfis[f'{ciri_name}_solidity'] = 0.0
            for i in range(7):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = 0.0
    ordered_feature_keys_100K_Depan = []
    for ciri_name in ciri_names_100K_Depan:
        ordered_feature_keys_100K_Depan.extend([
            f'{ciri_name}_area', f'{ciri_name}_aspect_ratio', f'{ciri_name}_solidity',
        ] + [f'{ciri_name}_hu_moment_{i}' for i in range(7)] + [ # 7 Hu Moments
            f'{ciri_name}_center_x_norm', f'{ciri_name}_center_y_norm',
            f'{ciri_name}_glcm_contrast', f'{ciri_name}_glcm_correlation',
            f'{ciri_name}_glcm_energy', f'{ciri_name}_glcm_homogeneity'
        ])
    #print(f"FINAL ANFIS INPUT VECTOR")
    final_anfis_input_vector = [final_features_for_anfis[key] for key in ordered_feature_keys_100K_Depan]
    #print(f"Ini adalah: {final_anfis_input_vector}")
    return np.array(final_anfis_input_vector)

def ekstraksi_bentuk_dan_pemberian_label_100K(final_segmentation_cleaned, cr_channel_masked):
    
    min_area_threshold=200
    max_area_threshold=20000
    final_segmentation_cleaned = final_segmentation_cleaned.astype(np.uint8) * 255
    # Temukan kontur dari objek-objek putih
    # RETR_EXTERNAL: Hanya mengambil kontur terluar (bagus untuk menghindari lubang internal sebagai kontur terpisah)
    contours, _ = cv2.findContours(final_segmentation_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labeled_objects_100k = {}
    #print(f"Ditemukan {len(contours)} kontur.") # Print jumlah kontur yang ditemukan
    
    # --- VISUALISASI KONTUR YANG DITEMUKAN ---
    # Buat salinan gambar hasil segmentasi untuk digambar
    # Kita akan menggambar kontur di atasnya untuk visualisasi
    kontur_image = np.zeros_like(final_segmentation_cleaned)
    # Jika Anda ingin melihat kontur di atas gambar segmentasi asli:
    kontur_image = final_segmentation_cleaned.copy()
    # Gambar semua kontur yang ditemukan dengan warna (misalnya hijau)
    # Menggambar dengan tebal 2 atau 3 agar terlihat jelas
    # Kontur akan digambar sebagai garis luar
    cv2.drawContours(kontur_image, contours, -1, (255, 255, 255), 2) # Menggambar di gambar hitam, kontur putih
    '''plt.figure(figsize=(12,5))
    plt.imshow(kontur_image, cmap='gray') # Gunakan cmap='gray' untuk gambar biner
    plt.title('Semua Kontur Ditemukan oleh findContours')
    plt.axis('off')
    plt.show()'''

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Dapatkan bounding box (x, y, lebar, tinggi)
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- VISUALISASI KONTUR SAAT INI DI ATAS GAMBAR SEGMENTASI ---
        # Buat salinan gambar basis untuk setiap iterasi agar hanya kontur saat ini yang disorot
        current_contour_display = cv2.cvtColor(kontur_image, cv2.COLOR_GRAY2BGR) # Konversi ke BGR untuk warna
        # Gambar kontur saat ini dengan warna cerah (misal: hijau)
        cv2.drawContours(current_contour_display, [contour], -1, (0, 255, 0), 2) # Warna hijau, tebal 2
        # Gambar bounding box juga bisa membantu
        cv2.rectangle(current_contour_display, (x, y), (x+w, y+h), (255, 0, 0), 2) # Warna biru, tebal 2
        
        '''plt.figure(figsize=(12,5))
        plt.imshow(current_contour_display) # imshow akan tahu ini BGR
        plt.title(f'Kontur {i} (Area: {area:.1f}) pada Segmentasi')
        plt.axis('off')
        plt.show() '''# Tampilkan setiap kontur satu per satu dengan highlight

        # Filter kontur berdasarkan area untuk menghilangkan noise dan sisa-sisa kecil
        if area < min_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu kecil: {area} < {min_area_threshold})")
            continue
        elif area > max_area_threshold:
            #print(f"Kontur {i} diabaikan (area terlalu besar: {area} > {max_area_threshold})")
            continue
        else:
            print(f"Kontur {i} ditemukan (Area: {area})")
        
        # Dapatkan ciri bentuk dasar
        aspect_ratio = float(w) / h if h != 0 else 0
        solidity = float(area) / (w * h) if (w*h) != 0 else 0
        # Contoh Hu Moments (untuk invarian rotasi, skala, translasi)
        # Hitung momen dulu
        M = cv2.moments(contour)
        if M["m00"] == 0: # Hindari pembagian nol jika momen area 0 (sangat kecil atau garis)
            print(f"Kontur {i} diabaikan (momen area nol).")
            continue
        hu_moments = cv2.HuMoments(M).flatten()
        # Hu moments seringkali diambil log untuk skala yang lebih baik
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        # Buat mask untuk objek individu
        object_mask = np.zeros_like(final_segmentation_cleaned)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)

        # --- LOGIKA IDENTIFIKASI/PEMBERIAN LABEL ---
        
        center_x_norm = (x + w/2) / final_segmentation_cleaned.shape[1]
        center_y_norm = (y + h/2) / final_segmentation_cleaned.shape[0]
        '''print(f"\tAspectRatio={aspect_ratio:.2f}, \n\tSolidity={solidity:.2f}")
        print(f"\tCenter Norm: ({center_x_norm:.2f}, {center_y_norm:.2f})")
        print(f"\tHu Moments (log): {hu_moments_log.tolist()}")'''

        label = "Lain-lain/Noise"
        # Logika untuk Angka Nominal (misal, 50000 atau 100000)
        # Biasanya di kanan bawah, area tertentu, rasio aspek tertentu
        if center_x_norm > 0.5 and center_y_norm > 0.6 and 3000 < area < 8000 and 2.0 < aspect_ratio < 6.0:
            label = "Angka Nominal"
        # Logika untuk Tulisan "BANK INDONESIA" (misal, di kanan atas, area tertentu, rasio aspek sangat tinggi)
        elif 0.5 < center_x_norm < 0.8 and center_y_norm < 0.4 and 300 < area < 750 and aspect_ratio > 2.0:
            label = "Bank"
        elif center_x_norm > 0.6 and center_y_norm < 0.4 and 700 < area < 1400 and aspect_ratio > 2.0:
            label = "Indonesia"
        elif center_x_norm > 0.5 and center_y_norm < 0.4 and 1300 < area < 3500 and aspect_ratio > 4.0:
            label = "Bank Indonesia"
        #print(f"\tLabel yang ditentukan: {label}")
        # Tambahkan ke dictionary jika itu salah satu ciri utama
        if label in ["Angka Nominal", "Bank", "Indonesia", "Bank Indonesia"] and label not in labeled_objects_100k:
            labeled_objects_100k[label] = {
                'mask': object_mask,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'hu_moments': hu_moments_log.tolist(),
                'center_x_norm': center_x_norm,
                'center_y_norm': center_y_norm } # Simpan sebagai list untuk kemudahan
                # tambahkan ciri bentuk lain di sini
            # Debug: Tampilkan objek yang teridentifikasi
           # print(f"Objek '{label}' berhasil ditambahkan.")
    final_features_for_anfis = {} # Dictionary untuk menyimpan semua fitur yang akan jadi input ANFIS 
    ciri_names_100K = ['Angka Nominal', 'Bank', 'Indonesia', 'Bank Indonesia'] # Tambahkan jika Anda punya label ini
    for ciri_name in ciri_names_100K:
        if ciri_name in labeled_objects_100k: # Pastikan ciri tersebut berhasil ditemukan dan dilabeli
            data_ciri = labeled_objects_100k[ciri_name]
            mask_ciri = data_ciri['mask'] # Ini dia mask untuk ciri spesifik!

            # --- MULAI BAGIAN BARU: EKSTRAKSI FITUR BENTUK DAN TEKSTUR ---
            
            # 1. Simpan Fitur Bentuk yang Sudah Dihitung
            # (Anda sudah memiliki ini dari langkah pelabelan sebelumnya)
            final_features_for_anfis[f'{ciri_name}_area'] = data_ciri['area']
            #print(f"FINAL FEATURES FOR ANFIS")
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = data_ciri['aspect_ratio']
            final_features_for_anfis[f'{ciri_name}_solidity'] = data_ciri['solidity']
            for i, hu in enumerate(data_ciri['hu_moments']):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = hu
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = data_ciri['center_x_norm']
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = data_ciri['center_y_norm']

            # 2. Ekstraksi Fitur Tekstur (GLCM)
            ciri_area_cr_masked = cv2.bitwise_and(cr_channel_masked, cr_channel_masked, mask=mask_ciri)
            # PENTING: Debug visual di sini untuk memastikan ciri_area_cr_masked terlihat benarprint(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            '''print(f"DEBUG INPUT GLCM: Min piksel: {np.min(ciri_area_cr_masked)}")
            print(f"DEBUG INPUT GLCM: Max piksel: {np.max(ciri_area_cr_masked)}")'''
            glcm_features = hitung_fitur_glcm(ciri_area_cr_masked)
            '''plt.figure(figsize=(12,5))
            plt.imshow(ciri_area_cr_masked, cmap='gray')
            plt.title(f'Masked GLCM Area for {ciri_name}')
            plt.show()'''

            #print(f"FINAL FEATURES FOR ANFIS GLCM")
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = glcm_features['glcm_contrast']
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = glcm_features['glcm_correlation']
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = glcm_features['glcm_energy']
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = glcm_features['glcm_homogeneity']
            #print(f"Ini adalah: {final_features_for_anfis}")
        else:
            # Jika ciri tidak ditemukan, isi dengan nilai default (misal 0.0)
            # Ini PENTING agar jumlah fitur selalu konsisten untuk ANFIS
            # Hitung berapa banyak fitur yang diharapkan untuk satu ciri (misal 5 bentuk + 7 Hu + 2 center + 4 GLCM = 18 fitur per ciri)
            num_features_per_ciri = 5 + 7 + 4 # sesuaikan dengan jumlah fitur yang Anda ekstrak
                
            # Buat placeholder untuk setiap fitur spesifik:
            final_features_for_anfis[f'{ciri_name}_area'] = 0.0
            final_features_for_anfis[f'{ciri_name}_aspect_ratio'] = 0.0
            final_features_for_anfis[f'{ciri_name}_solidity'] = 0.0
            for i in range(7):
                final_features_for_anfis[f'{ciri_name}_hu_moment_{i}'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_x_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_center_y_norm'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_contrast'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_correlation'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_energy'] = 0.0
            final_features_for_anfis[f'{ciri_name}_glcm_homogeneity'] = 0.0

    # Terakhir, ubah dictionary `final_features_for_anfis` menjadi array/list numerik
    # yang urutannya KONSISTEN. Ini sangat penting untuk ANFIS.
    # Anda perlu mendefinisikan urutan kunci fitur Anda.
    # Contoh urutan kunci untuk 50K:'''
    ordered_feature_keys_100K = []
    for ciri_name in ciri_names_100K:
        ordered_feature_keys_100K.extend([
            f'{ciri_name}_area', f'{ciri_name}_aspect_ratio', f'{ciri_name}_solidity',
        ] + [f'{ciri_name}_hu_moment_{i}' for i in range(7)] + [ # 7 Hu Moments
            f'{ciri_name}_center_x_norm', f'{ciri_name}_center_y_norm',
            f'{ciri_name}_glcm_contrast', f'{ciri_name}_glcm_correlation',
            f'{ciri_name}_glcm_energy', f'{ciri_name}_glcm_homogeneity'
        ])
    #print(f"FINAL ANFIS INPUT VECTOR")
    final_anfis_input_vector = [final_features_for_anfis[key] for key in ordered_feature_keys_100K]
    #print(f"Ini adalah: {final_anfis_input_vector}")

    return np.array(final_anfis_input_vector) # ANFIS biasanya butuh numpy array

# --- Fungsi untuk melakukan pemrosesan gambar ---
def proses_gambar(gambar, nama_file_original=" ", uang_type=None): 
    print(f"\n--- Memproses Gambar: {nama_file_original} ---")

    if gambar is None:
        print(f"ERROR: Gambar input ke proses_gambar adalah None untuk {nama_file_original}.")
        return None, None # PENTING: Mengembalikan DUA nilai None
    
    # --- CONTOH PEMROSESAN GAMBAR ---
    # 1. Konversi ke RGB
    gambar_RGB = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
    '''cv2.imshow('Format RGB', gambar_RGB)
    cv2.waitKey(0)'''

    # 1. Panggil fungsi untuk membuat mask uang kertas
    mask_uang_kertas = buat_mask_uang_kertas(gambar_RGB)

    mean_R = np.mean(gambar_RGB[:, :, 0]) # Rata-rata saluran Merah
    mean_G = np.mean(gambar_RGB[:, :, 1]) # Rata-rata saluran Hijau
    mean_B = np.mean(gambar_RGB[:, :, 2]) # Rata-rata saluran Biru

    '''print(f"\nRata-rata nilai RGB keseluruhan gambar:")
    print(f"  Rata-rata Merah (R): {mean_R:.2f}")
    print(f"  Rata-rata Hijau (G): {mean_G:.2f}")
    print(f"  Rata-rata Biru (B): {mean_B:.2f}")'''

    # --- MENAMPILKAN HASIL MASK UANG KERTAS SEBELUM MELANJUTKAN ---
    '''cv2.imshow('HASIL MASK UANG KERTAS (Tekan tombol apa saja untuk lanjut)', mask_uang_kertas)
    cv2.waitKey(0) # Tunggu sampai tombol ditekan
    cv2.destroyWindow('HASIL MASK UANG KERTAS (Tekan tombol apa saja untuk lanjut)')''' # Tutup jendela setelah tombol ditekan
    #print("Mask uang kertas ditampilkan. Melanjutkan proses...")
    # --- END MENAMPILKAN HASIL MASK ---

    # 2. Konversi ke YCbCr
    gambar_YCbCr = cv2.cvtColor(gambar, cv2.COLOR_BGR2YCrCb)
    #cv2.imshow('Format YCbCr', gambar_YCbCr)
    
    #print("Gambar diubah ke YCbCr.")
    #print("Tekan tombol apa saja untuk menutup jendela hasil pemrosesan.")
    '''cv2.waitKey(0) # Tunggu sampai tombol ditekan untuk menutup jendela hasil
    cv2.destroyAllWindows()'''

    # 3. Hitung Rata-Rata Tiap Saluran Y, Cb, dan Cr
    y_channel, cr_channel, cb_channel = cv2.split(gambar_YCbCr)
    mean_Y = np.mean(y_channel) 
    mean_Cr = np.mean(cr_channel) 
    mean_Cb = np.mean(cb_channel)

    '''print(f"\nRata-rata nilai YCbCr keseluruhan gambar:")
    print(f"  Rata-rata Luminance (Y): {mean_Y:.2f}")
    print(f"  Rata-rata Blue-difference Chroma (Cb): {mean_Cb:.2f}")
    print(f"  Rata-rata Red-difference Chroma (Cr): {mean_Cr:.2f}")'''

    if cr_channel is None or cr_channel.size == 0:
        print(f"ERROR: Saluran Cr tidak valid setelah persiapan untuk {nama_file_original}.")
        return None, None

    # 4. Terapkan Mask Uang Kertas ke Saluran Cr
    cr_channel_masked = cv2.bitwise_and(cr_channel, cr_channel, mask=mask_uang_kertas)
    #print("Saluran Cr dimask dengan mask uang kertas.")

    mean_Cr_masked = np.mean(cr_channel_masked)

    '''plt.figure(figsize=(12, 5))

    # Subplot 1: Saluran Y (grayscale)
    plt.subplot(1, 4, 1)
    plt.imshow(y_channel, cmap='gray')
    plt.title(f'Saluran Y (Rata-Rata: {mean_Y:.0f})')
    plt.axis('off')

    # Subplot 2: Saluran Cb (grayscale)
    plt.subplot(1, 4, 2)
    plt.imshow(cb_channel, cmap='gray')
    plt.title(f'Saluran Cb (Rata-Rata: {mean_Cb:.0f})')
    plt.axis('off')

    # Subplot 3: Saluran Cr (grayscale)
    plt.subplot(1, 4, 3)
    plt.imshow(cr_channel, cmap='gray')
    plt.title(f'Saluran Cr (Rata-Rata: {mean_Cr:.0f})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Saluran Cr setelah mask
    plt.figure(figsize=(12,5))
    plt.plot()
    plt.imshow(cr_channel_masked, cmap='gray')
    plt.title(f'Saluran Cr Setelah Mask (Rata-Rata: {mean_Cr_masked:.0f})')
    plt.axis('off')
    plt.show()

    # 4. Tampilkan HISTOGRAM untuk tiap saluran Y, Cb, Cr
    plt.figure(figsize=(12, 5))

    # Histogram Y Channel
    plt.subplot(1, 3, 1)
    plt.hist(y_channel.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.title('Histogram Saluran Y')
    plt.xlabel('Nilai Piksel')
    plt.ylabel('Frekuensi')
    plt.xlim([0, 256])

    # Histogram Cb Channel
    plt.subplot(1, 3, 2)
    # Cb dan Cr memiliki rentang yang berpusat di 128 (misal 16-240)
    # Anda bisa mengatur bins/range sesuai distribusi yang Anda harapkan
    plt.hist(cb_channel.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title('Histogram Saluran Cb')
    plt.xlabel('Nilai Piksel')
    plt.ylabel('Frekuensi')
    plt.xlim([0, 256])

    # Histogram Cr Channel
    plt.subplot(1, 3, 3)
    # Cb dan Cr memiliki rentang yang berpusat di 128 (misal 16-240)
    # Anda bisa mengatur bins/range sesuai distribusi yang Anda harapkan
    plt.hist(cr_channel.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title('Histogram Saluran Cr')
    plt.xlabel('Nilai Piksel')
    plt.ylabel('Frekuensi')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

    # Histogram Cr Channel (setelah masked)
    plt.plot()
    plt.hist(cr_channel_masked.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title('Histogram Saluran Cr Setelah Mask')
    plt.xlabel('Nilai Piksel')
    plt.ylabel('Frekuensi')
    plt.xlim([0, 256])

    #plt.tight_layout()
    plt.show()'''

    # --- Logika Thresholding Berdasarkan uang_type ---
    
    # Tentukan apakah ini uang tampak depan yang memerlukan looping thresholding
    #is_depan_uang_kertas = (uang_type == '50K_tampak depan' or uang_type == '100K_tampak depan')
    if uang_type == '50K_tampak depan':
        pecahan_teridentifikasi = identifikasi_pecahan_berdasarkan_cr_histogram(cr_channel_masked)
        '''print(f"DEBUG: Pecahan teridentifikasi: {pecahan_teridentifikasi}")
        print(f"Menggunakan thresholding untuk {uang_type}")'''
        
        threshold_50k_high = 100 
        threshold_50k_low = 20
        pola_ditemukan = False # Flag untuk menandakan pola patokan telah ditemukan
        _, th_bawah = cv2.threshold(cr_channel_masked, threshold_50k_low, 255, cv2.THRESH_BINARY)
        _, th_atas = cv2.threshold(cr_channel_masked, threshold_50k_high, 255, cv2.THRESH_BINARY_INV)

        final_segmentation = cv2.bitwise_and(th_bawah, th_atas)
        '''plt.figure(figsize=(12, 5))
        plt.plot()
        plt.imshow(final_segmentation, cmap='gray')
        plt.title(f'Thresholding untuk Uang {pecahan_teridentifikasi}')
        plt.axis('off')
        plt.show()'''

        kernel_clean_noise = np.ones((3,3),np.uint8)
        kernel_fill_holes = np.ones((3,3),np.uint8)
        final_segmentation_cleaned = cv2.morphologyEx(final_segmentation, cv2.MORPH_OPEN, kernel_clean_noise, iterations=2)
        final_segmentation_cleaned = cv2.morphologyEx(final_segmentation_cleaned, cv2.MORPH_CLOSE, kernel_fill_holes, iterations=2)
        '''plt.figure(figsize=(12, 5))
        plt.plot()
        plt.imshow(final_segmentation_cleaned, cmap='gray')
        plt.title(f'Thresholding Akhir untuk Uang {pecahan_teridentifikasi}')
        plt.axis('off')
        plt.show()'''

    elif uang_type == '100K_tampak depan':
        # Menggunakan parameter untuk 100 Ribu.
        #print("Menggunakan thresholding biasa untuk 100 Ribu.")
        # Tuning ambang batas ini! Lihat histogram Cr untuk 100K.
        # Nominal 100K terlihat TERANG di Cr, jadi pakai THRESH_BINARY.
        threshold_value_100k_high = 163 # Contoh: cari nilai di mana 100000 mulai putih
        #print(f" DEBUG: Menggunakan nilai threshold untuk {uang_type} = {threshold_value_100k_high}")
        _, final_segmentation = cv2.threshold(cr_channel_masked, threshold_value_100k_high, 255, cv2.THRESH_BINARY)
        '''plt.figure(figsize=(12, 5))
        plt.plot()
        plt.imshow(final_segmentation, cmap='gray')
        plt.title(f'Thresholding untuk Uang {uang_type}')
        plt.axis('off')
        plt.show()'''
        # ... (Pasca-pemrosesan morfologi) ...
        kernel_clean_noise = np.ones((3,3),np.uint8)
        kernel_fill_holes = np.ones((3,3),np.uint8)
        final_segmentation_cleaned = cv2.morphologyEx(final_segmentation, cv2.MORPH_OPEN, kernel_clean_noise, iterations=2)
        final_segmentation_cleaned = cv2.morphologyEx(final_segmentation_cleaned, cv2.MORPH_CLOSE, kernel_fill_holes, iterations=1)
        '''plt.figure(figsize=(12, 5))
        plt.plot()
        plt.imshow(final_segmentation_cleaned, cmap='gray')
        plt.title(f'Thresholding Akhir untuk Uang {uang_type}')
        plt.axis('off')
        plt.show()'''
    
    else: # Ini akan mencakup '50k_belakang' dan '100k_belakang'

        pecahan_teridentifikasi = identifikasi_pecahan_berdasarkan_cr_histogram(cr_channel_masked)
        #print(f"DEBUG: Pecahan teridentifikasi: {pecahan_teridentifikasi}")
        
        if pecahan_teridentifikasi == '100K':
            # Menggunakan parameter untuk 100 Ribu.
            #print("Menggunakan thresholding biasa untuk 100 Ribu.")
            # Tuning ambang batas ini! Lihat histogram Cr untuk 100K.
            # Nominal 100K terlihat TERANG di Cr, jadi pakai THRESH_BINARY.
            threshold_value_100k_high = 180 # Contoh: cari nilai di mana 100000 mulai putih
            #print(f" DEBUG: Menggunakan nilai threshold untuk {uang_type} = {threshold_value_100k_high}")
            _, final_segmentation = cv2.threshold(cr_channel_masked, threshold_value_100k_high, 255, cv2.THRESH_BINARY)
            '''plt.figure(figsize=(12, 5))
            plt.plot()
            plt.imshow(final_segmentation, cmap='gray')
            plt.title(f'Thresholding untuk Uang {pecahan_teridentifikasi}')
            plt.axis('off')
            plt.show()'''
            # ... (Pasca-pemrosesan morfologi) ...
            kernel_clean_noise = np.ones((3,3),np.uint8)
            kernel_fill_holes = np.ones((3,3),np.uint8)
            final_segmentation_cleaned = cv2.morphologyEx(final_segmentation, cv2.MORPH_OPEN, kernel_clean_noise, iterations=2)
            final_segmentation_cleaned = cv2.morphologyEx(final_segmentation_cleaned, cv2.MORPH_CLOSE, kernel_fill_holes, iterations=1)
            '''plt.figure(figsize=(12, 5))
            plt.plot()
            plt.imshow(final_segmentation_cleaned, cmap='gray')
            plt.title(f'Thresholding Akhir untuk Uang {pecahan_teridentifikasi}')
            plt.axis('off')
            plt.show()'''
            
        elif pecahan_teridentifikasi == '50K':
            #Menggunakan parameter untuk 50 Ribu.
            #print("Menggunakan thresholding biasa untuk 50 Ribu.")
            # Tuning ambang batas ini! Lihat histogram Cr untuk 100K.
            # Nominal 100K terlihat TERANG di Cr, jadi pakai THRESH_BINARY.
            threshold_50k_high = 100 
            threshold_50k_low = 20
            pola_ditemukan = False # Flag untuk menandakan pola patokan telah ditemukan
            _, th_bawah = cv2.threshold(cr_channel_masked, threshold_50k_low, 255, cv2.THRESH_BINARY)
            _, th_atas = cv2.threshold(cr_channel_masked, threshold_50k_high, 255, cv2.THRESH_BINARY_INV)

            final_segmentation = cv2.bitwise_and(th_bawah, th_atas)
            '''plt.figure(figsize=(12, 5))
            plt.plot()
            plt.imshow(final_segmentation, cmap='gray')
            plt.title(f'Thresholding untuk Uang {pecahan_teridentifikasi}')
            plt.axis('off')
            plt.show()'''

            kernel_clean_noise = np.ones((3,3),np.uint8)
            kernel_fill_holes = np.ones((3,3),np.uint8)
            final_segmentation_cleaned = cv2.morphologyEx(final_segmentation, cv2.MORPH_OPEN, kernel_clean_noise, iterations=2)
            final_segmentation_cleaned = cv2.morphologyEx(final_segmentation_cleaned, cv2.MORPH_CLOSE, kernel_fill_holes, iterations=2)
            '''plt.figure(figsize=(12, 5))
            plt.plot()
            plt.imshow(final_segmentation_cleaned, cmap='gray')
            plt.title(f'Thresholding Akhir untuk Uang {pecahan_teridentifikasi}')
            plt.axis('off')
            plt.show()'''
        
        else: # Pecahan tidak dapat diidentifikasi berdasarkan histogram
            print(f"ERROR: Pecahan belakang tidak dapat diidentifikasi untuk {nama_file_original}. Mengembalikan None, None.")
            return None, None
    if uang_type == 'tampak belakang': # Jika awalnya teridentifikasi sebagai 'tampak belakang' saja
        if pecahan_teridentifikasi == '50K':
            uang_type_for_features = '50K_tampak belakang'
        elif pecahan_teridentifikasi == '100K':
            uang_type_for_features = '100K_tampak belakang'
        else:
            # Jika identifikasi Cr gagal juga, maka uang_type_for_features tetap 'tampak belakang'
            # atau set ke None agar tidak diproses
            uang_type_for_features = None 
            print("Peringatan: Pecahan belakang tidak dapat diidentifikasi lebih lanjut untuk ekstraksi fitur.")
    else:
        # Jika bukan uang belakang, pakai uang_type yang sudah ada (misal 50K_tampak depan)
        uang_type_for_features = uang_type
    
    # --- Bagian Ekstraksi Fitur ---
    final_hasil_ekstraksi = None
    if uang_type_for_features == '50K_tampak depan':
        #print("INFO: Ekstraksi fitur untuk 50K tampak depan.")
        final_hasil_ekstraksi = ekstraksi_bentuk_dan_pemberian_label_50K_Depan(final_segmentation_cleaned, cr_channel_masked)
    elif uang_type_for_features == '100K_tampak depan':
        #print("INFO: Ekstraksi fitur untuk 100K tampak depan.")
        final_hasil_ekstraksi = ekstraksi_bentuk_dan_pemberian_label_100K_Depan(final_segmentation_cleaned, cr_channel_masked)
    elif uang_type_for_features == '50K_tampak belakang': # Kondisi baru ini
        #print("INFO: Ekstraksi fitur untuk 50K tampak belakang.")
        final_hasil_ekstraksi = ekstraksi_bentuk_dan_pemberian_label_50K(final_segmentation_cleaned, cr_channel_masked) 
    elif uang_type_for_features == '100K_tampak belakang': # Kondisi baru ini
        #print("INFO: Ekstraksi fitur untuk 100K tampak belakang.")
        final_hasil_ekstraksi = ekstraksi_bentuk_dan_pemberian_label_100K(final_segmentation_cleaned, cr_channel_masked)
    else:
        print(f"Peringatan: Tipe uang '{uang_type_for_features}' tidak dikenali untuk ekstraksi fitur. Skipping.")
        final_hasil_ekstraksi = None

    # Pastikan Anda mengembalikan final_ciri_khusus dari proses_gambar
    return final_hasil_ekstraksi
    
    
# --- Fungsi baru untuk menampilkan gambar dari folder ---
def tampilkan_dan_pilih_gambar_untuk_diproses(base_dataset_dir_for_selection):
    print("Menampilkan dan memilih gambar dari folder...")
    
    image_files = []
    for root, _, files in os.walk(base_dataset_dir_for_selection):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"Tidak ada gambar ditemukan di direktori dataset: {base_dataset_dir_for_selection}")
        return None
    print("\nDaftar Gambar yang Tersedia untuk Diproses:")
    for i, img_path in enumerate(image_files):
        print(f"{i+1}. {os.path.basename(img_path)} (Path: {os.path.relpath(img_path, base_dataset_dir_for_selection)})")
    while True:
        try:
            # PENTING: Di sini Anda memilih NOMOR, BUKAN menekan 'p'
            pilihan = int(input(f"Pilih nomor gambar (1-{len(image_files)}) atau 0 untuk batal: "))
            if pilihan == 0:
                return None
            elif 1 <= pilihan <= len(image_files):
                return image_files[pilihan - 1] # HANYA MENGEMBALIKAN PATH!
            else:
                print("Pilihan tidak valid. Silakan masukkan nomor yang benar.")
        except ValueError:
            print("Input tidak valid. Harap masukkan angka.")


# --- Fungsi utama untuk memproses satu gambar dan mengembalikan vektor fitur ---
'''def proses_gambar_dari_path(image_path):
    #print(f"\nMemproses gambar: {image_path}")
    # 1. Pemrosesan Gambar
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"ERROR: Gagal memuat gambar {image_path}")
        return None # Mengembalikan None jika gambar tidak bisa dimuat
    
    path_parts = image_path.lower().split(os.sep)
    path_parts_lower = [part.lower() for part in path_parts]
    print(f"path_parts: {path_parts}")
    print(f"path_parts_lower: {path_parts_lower}")

    uang_type = None
    if '50k' in path_parts_lower and 'depan' in path_parts_lower:
        uang_type = '50K_tampak depan'
    elif '100k' in path_parts_lower and 'depan' in path_parts_lower:
        uang_type = '100K_tampak depan'
    elif ('50k' in path_parts_lower or '100k' in path_parts_lower) and 'belakang' in path_parts_lower: # Perhatikan 'belakang' saja
        uang_type = 'tampak belakang'
    else:
        print(f"Peringatan: Tipe uang (50K/100K dan Depan/Belakang) tidak dapat dideteksi dari path: {image_path}. Skipping.")
        return None, None
    
    
    # Tentukan label gambar (asli=1, palsu=0)
    # Asumsi: Gambar asli ada di folder 'asli' dan palsu di folder 'palsu'
    label = None
    if 'asli' in path_parts:
        label = 1
    elif 'palsu' in path_parts:
        label = 0 # Atau -1, tergantung konvensi Anda
    else:
        print(f"Peringatan: Label (asli/palsu) tidak dapat dideteksi dari path: {image_path}. Skipping.")
        return None, None # Gagal menentukan label
    
    #print(f"INFO: Memproses gambar '{os.path.basename(image_path)}' sebagai '{uang_type}' dengan label '{label}'.")

    # Panggil fungsi proses_gambar dengan uang_type yang telah ditentukan
    final_anfis_input_vector = proses_gambar(original_image, os.path.basename(image_path), uang_type=uang_type)
    
    if final_anfis_input_vector is None:
        print(f"Gagal mengekstrak fitur untuk {os.path.basename(image_path)}.")
        return None, None
        
    #print(f"DEBUG: Fitur berhasil diekstraksi untuk {os.path.basename(image_path)}. Shape: {final_anfis_input_vector.shape}")
    
    return final_anfis_input_vector, label'''

def proses_gambar_dari_path(image_path):
    """
    Memproses gambar dari path file, mengekstrak fitur, dan mengembalikan vektor fitur dan label.
    """
    #print(f"\nMemproses gambar: {image_path}")
    
    # 1. Pemrosesan Gambar dan Validasi Awal
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"ERROR: Gagal memuat gambar {image_path}")
        return None, None # Mengembalikan None, None jika gambar tidak bisa dimuat
    
    path_parts = image_path.lower().split(os.sep)
    path_parts_lower = [part.lower() for part in path_parts]

    # 2. Tentukan Tipe Uang (50K/100K & Depan/Belakang)
    uang_type = None
    if '50k' in path_parts_lower and 'depan' in path_parts_lower:
        uang_type = '50K_tampak depan'
    elif '100k' in path_parts_lower and 'depan' in path_parts_lower:
        uang_type = '100K_tampak depan'
    elif '50k' in path_parts_lower and 'belakang' in path_parts_lower:
        uang_type = '50K_tampak belakang'
    elif '100k' in path_parts_lower and 'belakang' in path_parts_lower:
        uang_type = '100K_tampak belakang'
    else:
        print(f"Peringatan: Tipe uang (50K/100K dan Depan/Belakang) tidak dapat dideteksi dari path: {image_path}. Skipping.")
        return None, None # Gagal menentukan tipe uang
    
    # 3. Tentukan Label Gambar (asli=1, palsu=0)
    label = None
    if 'asli' in path_parts_lower:
        label = 1
    elif 'palsu' in path_parts_lower:
        label = 0
    else:
        print(f"Peringatan: Label (asli/palsu) tidak dapat dideteksi dari path: {image_path}. Skipping.")
        return None, None # Gagal menentukan label
    
    #print(f"INFO: Memproses gambar '{os.path.basename(image_path)}' sebagai '{uang_type}' dengan label '{label}'.")

    # 4. Panggil fungsi proses_gambar dengan uang_type yang telah ditentukan
    # ASUMSI: fungsi proses_gambar() mengembalikan 2 nilai (vektor fitur, dan label yang dihitung di dalam)
    # Ini mungkin tidak sesuai dengan desain Anda, jadi saya akan memanggilnya agar hanya mengembalikan fitur.
    final_anfis_input_vector = proses_gambar(original_image, os.path.basename(image_path), uang_type=uang_type)

    if final_anfis_input_vector is None:
        print(f"Gagal mengekstrak fitur untuk {os.path.basename(image_path)}.")
        return None, None
        
    # Pastikan hasil dari proses_gambar() adalah array NumPy
    if not isinstance(final_anfis_input_vector, np.ndarray):
        print(f"Peringatan: final_anfis_input_vector bukan numpy array. Mengkonversi...")
        final_anfis_input_vector = np.array(final_anfis_input_vector)

    # 5. Mengembalikan vektor fitur dan label yang ditentukan dari path
    return final_anfis_input_vector, label

    
if __name__ == "__main__":
    
    # >>>>>> KONFIGURASI <<<<<<
    base_dataset_dir = '/home/pi/MyEnv/Dataset Uang Kertas'
    output_features_dir = 'anfis_extracted_features' # Folder untuk menyimpan fitur
    os.makedirs(output_features_dir, exist_ok=True) # Buat folder jika belum ada

    # List untuk mengumpulkan semua vektor fitur dan label
    all_feature_vectors = []
    all_labels = []

    print("Pilih opsi:")
    print("1. Ambil Gambar dari Webcam (Live Preview)")
    print("2. Tampilkan dan Pilih Gambar dari Folder untuk Diproses")
    print("3. Proses SEMUA Gambar dari Folder (untuk membuat dataset)") # Opsi baru
    
    #pilihan_main_menu = input("Masukkan pilihan (1/2/3): ").strip()
    pilihan_main_menu = int(input("Masukkan pilihan (1/2/3): "))

    if pilihan_main_menu == 1:
        # --- OPSI 1: Ambil Gambar dari Webcam ---
        # Fungsi ini akan menangani semua input user dan menyimpan gambar
        file_path_tersimpan = ambil_gambar_dari_webcam(base_dataset_dir='Dataset Uang Kertas')

        if file_path_tersimpan:
            print(f"\nGambar berhasil disimpan di: {file_path_tersimpan}")
            print("Mulai proses ekstraksi fitur untuk gambar ini...")
            
            # Panggil proses_gambar_dari_path untuk gambar yang baru saja disimpan
            # Ini akan mengembalikan 2 nilai, pastikan kita menangani potensi None
            anfis_input_vector, label_anfis = proses_gambar_dari_path(file_path_tersimpan)

            if anfis_input_vector is not None:
                print(f"DEBUG: Fitur dari {os.path.basename(file_path_tersimpan)} (Label: {label_anfis}) berhasil diekstraksi.")
                print("Fitur:", anfis_input_vector)
            else:
                print(f"PERINGATAN: Gagal mengekstrak fitur ANFIS dari {os.path.basename(file_path_tersimpan)}.")
        else:
            print("Pengambilan gambar dibatalkan atau gagal.")

    elif pilihan_main_menu == 2:
        # --- OPSI 2: Tampilkan dan Pilih Gambar dari Folder untuk Diproses ---
        print("\n--- OPSI 2: Pemrosesan Gambar Individual dari Folder ---")
        dataset_base_dir_for_selection = 'Dataset Uang Kertas' 
        
        image_path_to_display = tampilkan_dan_pilih_gambar_untuk_diproses(dataset_base_dir_for_selection)

        if image_path_to_display: # Pengecekan penting jika user batal memilih (mengembalikan None)
            print(f"\nMemproses gambar yang dipilih: {image_path_to_display}")
            # DI SINI proses_gambar_dari_path dipanggil dengan PATH yang valid
            anfis_input_vector, label_anfis = proses_gambar_dari_path(image_path_to_display)
            
            if anfis_input_vector is not None:
                print(f"DEBUG: Fitur dari {os.path.basename(image_path_to_display)} (Label: {label_anfis}) berhasil diekstraksi.")
                print("Fitur:", anfis_input_vector)
            else:
                print(f"PERINGATAN: Gagal mengekstrak fitur ANFIS dari {os.path.basename(image_path_to_display)}.")
        else:
            print("Pemilihan gambar dibatalkan atau tidak ada gambar tersedia.")
        
    elif pilihan_main_menu == 3:
        # --- OPSI 3: Proses Semua Gambar dari Folder untuk Dataset ANFIS ---
        print("\n--- OPSI 3: Pemrosesan Seluruh Dataset untuk Ekstraksi Fitur ANFIS ---")
        dataset_base_dir_for_batch = 'Dataset Uang Kertas'
        if not os.path.isdir(dataset_base_dir_for_batch):
            print(f"ERROR: Direktori dataset untuk pemrosesan tidak ditemukan: {dataset_base_dir_for_batch}")
            print("Pastikan Anda sudah mengumpulkan gambar terlebih dahulu menggunakan opsi 1 atau perbarui 'dataset_base_dir'.")
        else:
            image_paths_to_process = []
            for root, _, files in os.walk(dataset_base_dir_for_batch):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths_to_process.append(os.path.join(root, file))
            if not image_paths_to_process:
                print(f"Tidak ada gambar ditemukan di direktori dataset: {dataset_base_dir_for_batch}")
            else:
                # --- Inisialisasi empat daftar terpisah untuk 4 jenis uang ---
                extracted_features_50k_depan = []
                extracted_labels_50k_depan = []
                extracted_features_50k_belakang = []
                extracted_labels_50k_belakang = []
                extracted_features_100k_depan = []
                extracted_labels_100k_depan = []
                extracted_features_100k_belakang = []
                extracted_labels_100k_belakang = []
                print(f"\nMemulai pemrosesan {len(image_paths_to_process)} gambar dari dataset...\n")

                for i, img_path in enumerate(image_paths_to_process):
                    # Memanggil fungsi proses_gambar_dari_path yang mengembalikan features dan label
                    features, label = proses_gambar_dari_path(img_path)
                    if features is not None and label is not None:
                        # --- DEBUGGING: CETAK PANJANG FITUR DAN INFORMASI ---
                        fitur_length = len(features)
                        path_parts_lower = [part.lower() for part in img_path.split(os.sep)]
                        # Mendapatkan uang_type dari path parts untuk logging
                        uang_type_for_logging = 'unknown'
                        if '50k' in path_parts_lower:
                            uang_type_for_logging = '50k'
                        elif '100k' in path_parts_lower:
                            uang_type_for_logging = '100k'
                        if 'depan' in path_parts_lower:
                            uang_type_for_logging += '_tampak depan'
                        elif 'belakang' in path_parts_lower:
                            uang_type_for_logging += '_tampak belakang'
                        print(f"[{i+1}/{len(image_paths_to_process)}] Memproses: {os.path.basename(img_path)} ({uang_type_for_logging})")
                        print(f"  -> Ekstraksi berhasil. Jumlah fitur: {fitur_length}")

                        # --- Logika pemisahan dataset di sini ---
                        # Logika ini harus konsisten dengan yang ada di proses_gambar_dari_path
                        if '50k' in path_parts_lower and 'depan' in path_parts_lower:
                            extracted_features_50k_depan.append(features)
                            extracted_labels_50k_depan.append(label)
                        elif '50k' in path_parts_lower and 'belakang' in path_parts_lower:
                            extracted_features_50k_belakang.append(features)
                            extracted_labels_50k_belakang.append(label)
                        elif '100k' in path_parts_lower and 'depan' in path_parts_lower:
                            extracted_features_100k_depan.append(features)
                            extracted_labels_100k_depan.append(label)
                        elif '100k' in path_parts_lower and 'belakang' in path_parts_lower:
                            extracted_features_100k_belakang.append(features)
                            extracted_labels_100k_belakang.append(label)
                    else:
                        print(f"[{i+1}/{len(image_paths_to_process)}] Skipping {os.path.basename(img_path)} karena gagal diproses atau tidak dikenali.")
                    
                print(f"\n--- Selesai memproses semua gambar dari dataset. ---")
            
                # --- Simpan empat dataset yang sudah dipisahkan ---
                output_features_dir = 'anfis_extracted_features'
                os.makedirs(output_features_dir, exist_ok=True)

                datasets_to_save = {
                '50k_depan': (extracted_features_50k_depan, extracted_labels_50k_depan),
                '50k_belakang': (extracted_features_50k_belakang, extracted_labels_50k_belakang),
                '100k_depan': (extracted_features_100k_depan, extracted_labels_100k_depan),
                '100k_belakang': (extracted_features_100k_belakang, extracted_labels_100k_belakang)}
                for name, (features_list, labels_list) in datasets_to_save.items():
                    if features_list:
                        try:
                            X = np.array(features_list)
                            y = np.array(labels_list)
                            np.save(os.path.join(output_features_dir, f'anfis_features_{name}.npy'), X)
                            np.save(os.path.join(output_features_dir, f'anfis_labels_{name}.npy'), y)
                            print(f"\nDataset '{name}' berhasil disimpan. (Shape X: {X.shape}, Shape Y: {y.shape})")
                        except ValueError as e:
                            print(f"\nERROR: Gagal membuat NumPy array untuk dataset '{name}'.")
                            print(f"Pesan error: {e}")
                            print("Ini berarti ada ketidakseragaman jumlah fitur dalam dataset ini. Periksa kembali log di atas untuk menemukan gambar yang bermasalah.")
                    else:
                        print(f"\nTidak ada gambar yang berhasil diproses untuk dataset '{name}'.")
    
    
    else:
        print("Pilihan tidak valid. Harap masukkan '1', '2', atau '3'.")


    '''elif pilihan_main_menu == 3:
        # --- OPSI 3: Proses Semua Gambar dari Folder untuk Dataset ANFIS ---
        print("\n--- OPSI 3: Pemrosesan Seluruh Dataset untuk Ekstraksi Fitur ANFIS ---")
        # Sesuaikan 'dataset_base_dir' ini dengan struktur folder yang dihasilkan oleh 'ambil_gambar_dari_webcam'
        dataset_base_dir_for_batch = 'Dataset Uang Kertas' 

        if not os.path.isdir(dataset_base_dir_for_batch):
            print(f"ERROR: Direktori dataset untuk pemrosesan tidak ditemukan: {dataset_base_dir_for_batch}")
            print("Pastikan Anda sudah mengumpulkan gambar terlebih dahulu menggunakan opsi 1 atau perbarui 'dataset_base_dir'.")
        else:
            image_paths_to_process = []
            for root, _, files in os.walk(dataset_base_dir_for_batch):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_paths_to_process.append(os.path.join(root, file))

            if not image_paths_to_process:
                print(f"Tidak ada gambar ditemukan di direktori dataset: {dataset_base_dir_for_batch}")
            else:
                extracted_features = []
                extracted_labels = []

                print(f"\nMemulai pemrosesan {len(image_paths_to_process)} gambar dari dataset...\n")

                for img_path in image_paths_to_process:
                    # Pastikan proses_gambar_dari_path mengembalikan 2 nilai atau 2 None
                    features, label = proses_gambar_dari_path(img_path)
                    if features is not None and label is not None:
                        extracted_features.append(features)
                        extracted_labels.append(label)
                    else:
                        print(f"Skipping {os.path.basename(img_path)} karena gagal diproses atau tidak dikenali.")
                print(f"Ini adalah: {extracted_features}")
                print(f"Ini adalah: {extracted_labels}")

                if extracted_features:
                    X_dataset = np.array(extracted_features)
                    y_dataset = np.array(extracted_labels)

                    output_features_dir = 'anfis_extracted_features'
                    os.makedirs(output_features_dir, exist_ok=True)
                    features_filepath = os.path.join(output_features_dir, 'anfis_features.npy')
                    labels_filepath = os.path.join(output_features_dir, 'anfis_labels.npy')

                    np.save(features_filepath, X_dataset)
                    np.save(labels_filepath, y_dataset)
                    print(f"\nPROSES SELESAI: Dataset fitur berhasil disimpan.")
                    print(f"File Fitur (X): {features_filepath} (Shape: {X_dataset.shape})")
                    print(f"File Label (Y): {labels_filepath} (Shape: {y_dataset.shape})")
                    print(f"Isi Dataset Label (Y): \n{y_dataset}") #
                else:
                    print("Tidak ada gambar yang berhasil diproses untuk membentuk dataset fitur ANFIS.")'''