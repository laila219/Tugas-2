"""
==========================================================================
 PERCOBAAN 2 — ROTASI SUDUT BEBAS
 Modul 2: Pembentukan Citra (Image Formation)

 Tujuan  : Merotasi gambar dengan sudut bebas menggunakan matriks rotasi.
 Konsep  : cv2.getRotationMatrix2D(center, angle, scale)
           cv2.warpAffine(src, M, (w, h))
           Rotasi 2D: R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(SCRIPT_DIR, "image")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rotasi_sederhana(img, sudut, scale=1.0):
    """Rotasi gambar pada pusat dengan sudut tertentu."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)  # Pusat rotasi = tengah gambar

    # ★ INTI: Buat matriks rotasi 2×3
    # sudut (derajat): positif=berlawanan jarum jam, negatif=searah jarum jam
    # scale: faktor zoom bersamaan saat rotasi (1.0 = tidak zoom)
    # Hasilnya: matriks 2×3 berisi cosθ, sinθ, dan offset translasi
    M = cv2.getRotationMatrix2D(center, sudut, scale)

    # ★ INTI: Terapkan matriks rotasi ke gambar
    # Ukuran output sama (w, h) → pojok gambar yang melebihi dimensi akan terpotong
    # Area kosong yang muncul (di pojok) → diisi hitam
    hasil = cv2.warpAffine(img, M, (w, h))
    print(f"  Rotasi {sudut}° (scale={scale})")
    return hasil


def rotasi_tanpa_crop(img, sudut):
    """Rotasi gambar dengan canvas yang diperbesar agar tidak terpotong."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, sudut, 1.0)

    # ★ INTI: Hitung ukuran canvas baru agar seluruh gambar muat setelah rotasi
    # cos & sin dari matriks rotasi digunakan untuk menghitung lebar/tinggi baru
    # Tanpa ini, pojok gambar akan terpotong saat sudut bukan kelipatan 90°
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    w_baru = int(h * sin_val + w * cos_val)  # lebar canvas baru
    h_baru = int(h * cos_val + w * sin_val)  # tinggi canvas baru

    # ★ INTI: Sesuaikan offset translasi agar gambar tetap berada di tengah canvas
    # Tanpa penyesuaian ini, gambar akan keluar dari canvas baru
    M[0, 2] += (w_baru - w) / 2
    M[1, 2] += (h_baru - h) / 2

    hasil = cv2.warpAffine(img, M, (w_baru, h_baru))
    print(f"  Rotasi {sudut}° tanpa crop: {w}×{h} → {w_baru}×{h_baru}")
    return hasil


def rotasi_multi_sudut(img, sudut_list):
    """Rotasi gambar ke beberapa sudut sekaligus."""
    return [(s, rotasi_sederhana(img, s)) for s in sudut_list]


def tampilkan_hasil(img, rotasi_list, rotasi_nocrop):
    """Visualisasi rotasi berbagai sudut."""
    n = len(rotasi_list)
    fig, axes = plt.subplots(2, max(n, 3), figsize=(4 * n, 8))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    for i, (s, im) in enumerate(rotasi_list):
        if i + 1 < axes.shape[1]:
            axes[0, i + 1].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[0, i + 1].set_title(f"Rotasi {s}°")
            axes[0, i + 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original"); axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(rotasi_nocrop, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("30° Tanpa Crop"); axes[1, 1].axis("off")
    for j in range(2, axes.shape[1]):
        axes[1, j].axis("off")

    plt.suptitle("Percobaan 2 — Rotasi Sudut Bebas", fontweight="bold")
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "02_rotasi_sudut_bebas.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[SIMPAN] {out}")


def main():
    print("=" * 60)
    print(" PERCOBAAN 2: ROTASI SUDUT BEBAS")
    print("=" * 60)

    img = cv2.imread(os.path.join(IMAGE_DIR, "kuda.jpg"))
    print(f"\n  Ukuran gambar: {img.shape}")

    print("\n[1] Rotasi berbagai sudut:")
    # Rotasi ke 4 sudut: 45°, 90°, 135°, 180°
    # Amati: semakin besar sudut, semakin banyak area hitam di pojok
    sudut_list = [45, 90, 135, 180]
    rotasi_list = rotasi_multi_sudut(img, sudut_list)

    print("\n[2] Rotasi tanpa crop:")
    # Rotasi 30° dengan canvas diperbesar → tidak ada bagian yang terpotong
    r_nocrop = rotasi_tanpa_crop(img, 30)

    tampilkan_hasil(img, rotasi_list, r_nocrop)

    print("\nRINGKASAN:")
    print("  getRotationMatrix2D(center, angle, scale) → matriks 2×3")
    print("  warpAffine(img, M, (w,h)) → terapkan rotasi")
    print("  Rotasi tanpa crop: perbesar canvas sesuai sin/cos sudut")


if __name__ == "__main__":
    main()


# ==========================================================================
# ANALISIS PERCOBAAN 1–2: TRANSLASI DAN ROTASI
# ==========================================================================
#
# Q1: Jelaskan mengapa muncul area hitam setelah translasi dan rotasi.
# A : Saat translasi atau rotasi diterapkan, piksel sumber tidak selalu
#     memetakan ke seluruh area gambar output. Piksel output yang tidak
#     memiliki pasangan di gambar sumber akan diisi nilai default (0 = hitam)
#     oleh cv2.warpAffine. Semakin besar pergeseran/sudut, semakin luas
#     area hitam yang muncul di tepi gambar.
#
# Q2: Apa solusi agar gambar tidak terpotong saat dirotasi 45°?
# A : Perbesar ukuran canvas output menggunakan rumus:
#       w_baru = int(abs(w*cosθ) + abs(h*sinθ))
#       h_baru = int(abs(w*sinθ) + abs(h*cosθ))
#     Kemudian tambahkan offset translasi pada M[0,2] dan M[1,2] agar
#     gambar tetap berada di tengah canvas yang lebih besar.
#     Fungsi rotasi_tanpa_crop() dalam percobaan ini mengimplementasikannya.
#
# Q3: Bagaimana center of rotation mempengaruhi hasil?
# A : Center of rotation adalah titik yang tidak bergerak selama rotasi.
#     Jika pusat di (w/2, h/2), gambar berputar di tempat tanpa translasi.
#     Jika pusat digeser ke pojok gambar, output akan tampak berputar
#     sekeliling pojok tersebut sehingga gambar berpindah jauh dari posisi
#     semula. Memilih pusat yang tepat sangat penting untuk aplikasi seperti
#     panorama stitch atau image alignment.
# ==========================================================================
