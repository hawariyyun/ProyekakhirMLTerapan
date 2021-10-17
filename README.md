# ProyekakhirMLTerapan
# Data Diri

Rifki Ramadani <br>
Proyek Akhir Machine Learning Terapan <br>
Kabupaten Pasaman, Sumatera Barat <br>

# Domain Masalah

Sistem rekomendasi merupakan algoritma pemrograman yang populer digunakan pada saat ini, algoritma yang bekerja untuk merekomendasikan produk-produk maupun layanan-layanan yang tersedia kepada pelanggan atau pengguna layanan. Sistem rekomendasi memiliki tujuan meningkatkatkan kuantitas produk yang dapat dijual dengan mengetahui minat dari pengguna tanpa mengurangi rasa puas yang dapat diberikan dari rekomendasi yang diberikan.\
Sistem rekomendasi dapat diterapkan pada berbagai layanan seperti sistem rekomendasi hotel, restoran, _e-commerce_, musik, dan sebagainya.\
Contoh penerapan sistem rekomendasi menurut (Luh, Ni G.P.S, 2017) diterapkan pada rekomendasi pemilihan mobil menggunakan algoritma K-Nearest Neighbor dengan beberapa parameter seperti rentang harga, tujuan beli, kapasitas, dan sebagainya. Namun algoritma ini dapat dikembangkan dengan kombinasi menggunakan algoritma lain.
\
(Badriyah, Tessy dkk., 2018) menerapkan sistem rekomendasi berbasis penyaringan konten (_content based filtering_) pada _e-commerce_ dengan menganalisis pola-pola yang ada untuk meningkatkan kepuasan pelanggan dan keuntungan penjual termasuk _e-commerce_ itu sendiri
\
Algoritma dalam sistem rekomendasi yang dipersonalisasi ada tiga jenis yaitu berbasis konten, kolaboratif dan hibrid. Pada pembahasan ini akan mengulas algoritma kolaboratif menggunakan metode berbasis model yaitu *cluster based algorithm*. Baik atau buruknya sistem rekomendasi pada kenyataannya dapat dinilai dengan pengguna yang menerima rekomendasi dan menolak rekomendasinya, namun pada pembelajaran mesin (*machine learning*) baik atau tidaknya suatu sistem rekomendasi dalam merekomendasikan sesuatu dapat diukur dengan metrik-metrik evaluasi yang tersedia.

**Referensi** : 


1.   Luh, Ni G.P.S.2017.Implementation of K-Nearest Neighbor Method for Car Selection Recommendation System._Techno.COM_.16(2).p.120-131.
2.   Badriyah, Dkk.2018.Sistem Rekomendasi Content Based Filtering Menggunakan
Algoritma Apriori.Konferensi Nasional Sistem Informasi-2018

# Permasalahan

Sistem rekomendasi pada penjelasan di atas dapat diterapkan pada berbagai bidang, salah satunya musik. Musik yang tersedia pada berbagai platform penyedia layanan musik pada contohnya Spotify. Sistem rekomendasi pada pada spotify dapat dibangun menggunakan metode-metode beragam dengan salah satu contohnya yaitu metode *cluster based algorithm* menggunakan *KMeans clustering*. Sehingga dapat diusulkan permasalahan sebagai berikut :

1.   Bagaimana algoritma KMeans clustering bekerja pada sistem rekomendasi musik pada data musik spotify?
2.   Bagaimana hasil rekomendasi musik berdasarkan pengelompokkan KMeans?

Untuk menjawab permasalahan dapat dibangun sebuah sistem rekomendasi dengan tujuan sebagai berikut :
1.   Mengetahui cara kerja dari algoritma KMeans Clustering dalam merekomendasikan musik.
2.   Membuat sistem rekomendasi musik dan mengetahui perbandingan parameter pembangun pada sistem rekomendasi yang dibuat.

## Metodologi

Sistem rekomendasi musik pada spotify dataset ini menggunakan metode berbasis model _cluster based algorithm_ menggunakan algoritma pengelompokkan KMeans.

## Matriks Evaluasi

Metrik yang digunakan dalam evaluasi sistem rekomendasi ini adalah metrik skor silhouette, metrik ini digunakan karena sistem rekomendasi dibangun menggunakan algoritma pengelompokkan bukan menggunakan algoritma prediksi. Metrik skor silhouette menghitung seberapa baiknya algoritma dalam mengelompokkan data dengan nilai terbaik 1 dan terburuk -1.

# Pemahaman Data

Sumber data : https://www.kaggle.com/accountstatus/spotify-songs-eda-andrecommendation-system/data?select=genres_v2.csv

---
Dataset diunduh dan dipindahkan ke repositori github. \
Dataset yang digunakan merupakan dataset yang terdiri dari kumpulan data lagu dari platform spotify 

## Penjelasan Atribut

Fitur-fitur pada dataset spotifyDataset.csv
*   danceability : mendeskripsikan kecocokan musik dengan tarian 0(least danceable)-1(most danceable)
*   energy : 0-1 pengukuran persepsi aktivitas dan intensitas, musik cepat, keras, ataupun berantakan
*   key : kunci pada musik, dipetakan dengan integer 0=C, 1=C# dsb.
*   loudness : intensitas suara yang ditimbulkan musik (dB)
*   mode : modalitas (mayor atau minor) 0/1
*   spechiness : deteksi kehadiran pengucapan pada track lagu
*   accousticness : pengukuran seberapa akustik musik (0-1)
*   instrumentalas : prediksi apakah sebuah track lagu tidak mengandung vokal
*   livesness : deteksi apakah musik dibawakan pada pertunjukan langsung atau merupakan rekaman
*   valence : positifitas musik (senang/sedih)
*   tempo : estimasi tempo (beat per minute)
*   type : tipe musik yang disediakan
*   id : identitas musik
*   uri : link ke musik pada spotify
*   track_href : link ke track
*   analysis_url : link untuk analysis
*   duration_ms : durasi dari satu musik(satu track musik) dalam milidetik
*   time_signature : estimasi waktu dalam satu track musik
*   genre : genre lagu
*   song_name : judul lagu

## Import Library

Import library-library yang dibutuhkan, seperti kebutuhan membaca data, visualisasi, preprocessing, membuat model dan sebagainya.
## Membaca Dataset

Baca dataset yang sudah dipindahkan ke repositori github menggunakan command : \
!wget https://raw.githubusercontent.com/hawariyyun/Proyek-ML-Terapan/main/spotifyDataset.csv \
menyimpan dataset ke dalam variabel df \
melihat sebaran dataset menggunakan fungsi describe() \
melihat ukuran dataset menggunakan fungsi shape

## Visualisasi awal
Membuat variabel cols yang berisikan list dari kolom ke 11 sampai akhir dataset dan menghapus kolom genre \
menyalin dataset dan menyimpannya ke dalam variabel data \
mengatur gaya visualisasi \
Visualisasi data menjelaskan bagaimana pengelompokkan genre musik yang ada terhadap fitur \
![visualisasi genre](https://user-images.githubusercontent.com/82726099/137630224-761a545f-b589-4e27-b877-d7ab3fc570b8.png)
Visualisasi sebaran data durasi musik terhadap genre 

# Persiapan Data

## Pengecekan null data dan Menghapus Kolom yang tidak diperlukan

Melihat apakah terdapat data kosong pada dataset dengan menggunakan fungsi isnull() dan mengembalikan jumlah data kosong tersebut pada output \
Menghapus kolom Unnamed: 0 dan title menggunakan fungsi drop(). Menghapus data kosong pada kolom song_name menggunakan fungsi dropna()

## Normalisasi menggunakan MinMaxScaler

Ambil tipe data numerik untuk normalisasi \
Normalisasi menggunakan minmax secara manual, hal ini bertujuan untuk memastikan data berada pada rentang nilai 0 hingga 1 \
Hasil normalisasi menggunakan minmax dapat dilihat seperti berikut, dataset bertipe numerik berada pada rendang nilai 0 hingga 1.

# Membangun Model

## Membangun model pengelompokkan KMeans
Algoritma KMeans dibangun dengan parameter n_cluster = 5, max_iter=1000 dengan maksud iterasi yang dilakukan tidak melebihi 1000 iterasi.

## Instansiasi kelas SpotifyRecommenderMHT, SpotifyRecommenderED, dan Spotify RecommenderCS

1. Kelas rekomendasi menggunakan jarak manhattan
Proses yang terjadi adalah : 
* Membaca data masukan
* Pemisahan data recurrent dan result
* Pengukuran jarak manhattan
* Menyimpan jarak pada variabel distance dan disimpan pada data result
* Mengembalikan nilai distance dan data result dengan kolom genre dan nama lagu

2. Kelas rekomendasi menggunakan jarak euclidian
Proses yang terjadi adalah : 
* Membaca data masukan
* Pemisahan data recurrent dan result
* Pengukuran jarak euclidian
* Menyimpan jarak pada variabel distance dan disimpan pada data result
* Mengembalikan nilai distance dan data result dengan kolom genre dan nama lagu

3. Kelas rekomendasi menggunakan cosine similarity
Proses yang terjadi adalah : 
* Membaca data masukan
* Pemisahan data recurrent dan result
* Pengukuran cosine similarity
* Menyimpan jarak pada variabel distance dan disimpan pada data result
* Mengembalikan nilai distance dan data result dengan kolom genre dan nama lagu

## Hasil rekomendasi masing-masing kelas rekomendasi

1. Hasil rekomendasi kelas rekomendasi manhattan \
![recMh](https://user-images.githubusercontent.com/82726099/137630741-7e771a70-fd48-4c34-b32a-21e72e61936f.PNG)
2. Hasil rekomendasi kelas rekomendasi euclidian \
![recEd](https://user-images.githubusercontent.com/82726099/137630754-6c6f90c4-e174-437e-99b6-b7afe255e311.PNG)
3. Hasil rekomendasi kelas rekomendasi cosine similarity \
![recMh](https://user-images.githubusercontent.com/82726099/137630781-68e97985-e5c0-4dc3-a464-5dba0df2166a.PNG)

# Evaluasi

Evaluasi *KMeans clustering* pada model menggunakan metrik silhouette_score dari library sklearn.metrics. Metrik silhouette_score mengevaluasi seberapa baiknya algoritma pengelompokkan(_clustering_) dalam mengelompokkan data. \
Nilai silhouette_score terbaik adalah 1 yang mana menunjukkan data terkelompokkan secara sempurna, sedangkan nilai terburuk adalah -1 yang mana data dikelompokkan pada kelompok yang secara total salah. \
Negatif atau tanda '-' menunjukkan pengelompokkan data yang dikelompokkan pada kelompok yang salah \
\
Berikut merupakan nilai silhouette_score pada masing-masing sistem rekomendasi musik menggunakan pengukuran kemiripan berbeda:
*   Euclidian : 0.5276831291369836
*   Manhattan : 0.5271553435109945
*   Cosine    : -0.0821256460960992

Evaluasi cosine_similarity terdapat tanda negatif yang mana merekomendasikan lagu dengan tingkat kemiripan rendah dan  kelompok lagu yang salah
