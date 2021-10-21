# Laporan Proyek Machine Learning Terapan - Rifki Ramadani
## Project Overview

Spotify merupakan platform penyedia musik populer pada masa sekarang, menyediakan musik dengan berbagai genre musik. Genre-genre musik yang ada tentunya berpengaruh dalam menentukan minat musik dari pengguna, misalkan seorang pengguna menyukai musik dengan genre pop saja, tentunya platform penyedia musik akan memberikan musik dengan genre sesuai dengan yang disukai oleh pengguna tersebut. Musik dengan genre yang memiliki kemiripan yang sama dapat diberikan kepada pengguna platform dengan menggunakan sistem rekomendasi. Sistem rekomendasi merupakan sistem yang merekomendasikan suatu hal berupa produk dalam rangka meningkatkan kuantitas penjualan produk dan kepuasaan pengguna layanan.\

Sistem rekomendasi dapat diterapkan pada berbagai layanan seperti sistem rekomendasi hotel, restoran, _e-commerce_, musik, dan sebagainya.\
Contoh penerapan sistem rekomendasi menurut (Luh, Ni G.P.S, 2017) diterapkan pada rekomendasi pemilihan mobil menggunakan algoritma K-Nearest Neighbor dengan beberapa parameter seperti rentang harga, tujuan beli, kapasitas, dan sebagainya. Namun algoritma ini dapat dikembangkan dengan kombinasi menggunakan algoritma lain.
\
(Badriyah, Tessy dkk., 2018) menerapkan sistem rekomendasi berbasis penyaringan konten (_content based filtering_) pada _e-commerce_ dengan menganalisis pola-pola yang ada untuk meningkatkan kepuasan pelanggan dan keuntungan penjual termasuk _e-commerce_ itu sendiri
\
Sistem rekomendasi pada dunia musik dalam merekomendasikan musik berdasarkan genre, dapat diterapkan pada data dari platform Spotify yang telah diuraikan sebelumnya. Sistem rekomendasi ini diharapkan dapat membantu meningkatkan kepuasaan pelanggan dengan harapan rekomendasi yang sesuai target, menigkatkan kuantitas penjualan/pendengar musik dari segi penyedia musik, dan diharapkan dapat meningkatkan pemakaian platform Spotify karena berkaitan erat dengan kepuasan pelanggan.

**Referensi** : 


1.   Luh, Ni G.P.S.2017.Implementation of K-Nearest Neighbor Method for Car Selection Recommendation System._Techno.COM_.16(2).p.120-131.
2.   Badriyah, Dkk.2018.Sistem Rekomendasi Content Based Filtering Menggunakan
Algoritma Apriori.Konferensi Nasional Sistem Informasi-2018

### Business Understanding

Sistem rekomendasi pada penjelasan di atas dapat diterapkan pada berbagai bidang, salah satunya musik. Musik yang tersedia pada berbagai platform penyedia layanan musik pada contohnya Spotify. Sistem rekomendasi pada pada spotify dapat dibangun menggunakan metode-metode beragam dengan salah satu contohnya yaitu metode *cluster based algorithm* menggunakan *KMeans clustering*. Sehingga dapat diusulkan permasalahan sebagai berikut :
### Problem Statement
1.   Bagaimana algoritma KMeans clustering bekerja pada sistem rekomendasi musik pada data musik spotify?
2.   Bagaimana hasil rekomendasi musik berdasarkan pengelompokkan KMeans?
### Goals
Untuk menjawab permasalahan dapat dibangun sebuah sistem rekomendasi dengan tujuan sebagai berikut :
1.   Mengetahui cara kerja dari algoritma KMeans Clustering dalam merekomendasikan musik.
2.   Membuat sistem rekomendasi musik dan mengetahui perbandingan parameter pembangun pada sistem rekomendasi yang dibuat.

##% Solution Approach
Solusi yang diterapkan agar sistem rekomendasi musik pada spotify dataset ini dapat berjalan dengan baik menggunakan sistem rekomendasi _collaborative based filtering_ dengan metode berbasis model _cluster based algorithm_ menggunakan algoritma pengelompokkan KMeans. \
Metrik yang digunakan dalam evaluasi sistem rekomendasi ini adalah metrik skor silhouette, metrik ini digunakan karena sistem rekomendasi dibangun menggunakan algoritma pengelompokkan bukan menggunakan algoritma prediksi. Metrik skor silhouette menghitung seberapa baiknya algoritma dalam mengelompokkan data dengan nilai terbaik 1 dan terburuk -1.

## Data Understanding

Dataset yang digunakan merupakan dataset yang menjelaskan fitur-fitur musik pada platform Spotify, berasal dari  : https://www.kaggle.com/accountstatus/spotify-songs-eda-andrecommendation-system/data?select=genres_v2.csv

---

Penjelasan Atribut \

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

Pada tahapan ini dilakukan _importing libraries_ yang digunakan untuk menyelesaikan tahapan demi tahapan pada penyelesaian masalah sistem rekomendasi ini. Setelah _import library_ dilakukan pembacaan dataset yang digunakan untuk analisis dan melakukan rekomendasi.

### Visualisasi awal
Membuat variabel cols yang berisikan list dari kolom ke 11 sampai akhir dataset dan menghapus kolom genre \
menyalin dataset dan menyimpannya ke dalam variabel data \
mengatur gaya visualisasi \
Visualisasi data menjelaskan bagaimana pengelompokkan genre musik yang ada terhadap fitur \
![visualisasi genre](https://user-images.githubusercontent.com/82726099/137630224-761a545f-b589-4e27-b877-d7ab3fc570b8.png)
Visualisasi sebaran data durasi musik terhadap genre 

## Data Preparation

### Pengecekan null data dan Menghapus Kolom yang tidak diperlukan

Melihat apakah terdapat data kosong pada dataset dengan menggunakan fungsi isnull() dan mengembalikan jumlah data kosong tersebut pada output \
Menghapus kolom Unnamed: 0 dan title menggunakan fungsi drop(). Menghapus data kosong pada kolom song_name menggunakan fungsi dropna()

### Normalisasi menggunakan MinMaxScaler

Normalisasi menggunakan minmax secara manual, hal ini bertujuan untuk memastikan data berada pada rentang nilai 0 hingga 1 \
Hasil normalisasi menggunakan minmax dapat dilihat seperti berikut, dataset bertipe numerik berada pada rendang nilai 0 hingga 1.

## Modelling and Result

### Membangun model pengelompokkan KMeans
Tahapan ini merupakan tahapan awal untuk membangun sistem rekomendasi, Algoritma yang digunakan untuk membangun model adalah algoritma KMeans dibangun dengan parameter n_cluster = 5, max_iter=1000 dengan maksud iterasi yang dilakukan tidak melebihi 1000 iterasi.

### Instansiasi kelas SpotifyRecommenderMHT, SpotifyRecommenderED, dan Spotify RecommenderCS

Kelas-kelas yang diinstansiasi merupakan kelas-kelas yang berguna untuk menampilkan hasil rekomendasi berdasarkan perhitungan jarak dari hasil kelas pengelompokkan oleh KMeans dan menggunakan perhitungan jarak berbeda pada masing-masing kelasnya.
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

### Hasil rekomendasi masing-masing kelas rekomendasi
Hasil rekomendasi diperoleh dengan mencari rekomendasi pada masing-masing rekomender menggunakan data judul musik dan genre musik yang sama pada masing-masing kelas rekomendasi.
1. Hasil rekomendasi kelas rekomendasi manhattan dengan 5 rekomendasi terbaik\
![recMh](https://user-images.githubusercontent.com/82726099/137630741-7e771a70-fd48-4c34-b32a-21e72e61936f.PNG)
2. Hasil rekomendasi kelas rekomendasi euclidian dengan 5 rekomendasi terbaik \
![recEd](https://user-images.githubusercontent.com/82726099/137630754-6c6f90c4-e174-437e-99b6-b7afe255e311.PNG)
3. Hasil rekomendasi kelas rekomendasi cosine similarity dengan 5 rekomendasi \
![recMh](https://user-images.githubusercontent.com/82726099/137630781-68e97985-e5c0-4dc3-a464-5dba0df2166a.PNG)

## Evaluation

Evaluasi collaborative filtering berbasi model dengan metode _cluster based algorithm_ menggunakan algoritma klustering *KMeans clustering* pada model menggunakan metrik silhouette_score dari library sklearn.metrics. Metrik silhouette_score mengevaluasi seberapa baiknya algoritma pengelompokkan(_clustering_) dalam mengelompokkan data. \
Nilai silhouette_score terbaik adalah 1 yang mana menunjukkan data terkelompokkan secara sempurna, sedangkan nilai terburuk adalah -1 yang mana data dikelompokkan pada kelompok yang secara total salah. \
Negatif atau tanda '-' menunjukkan pengelompokkan data yang dikelompokkan pada kelompok yang salah \
\
Berikut merupakan nilai silhouette_score pada masing-masing sistem rekomendasi musik menggunakan pengukuran kemiripan berbeda:
*   Euclidian : 0.5276831291369836
*   Manhattan : 0.5271553435109945
*   Cosine    : -0.0821256460960992

Evaluasi cosine_similarity terdapat tanda negatif yang mana merekomendasikan lagu dengan tingkat kemiripan rendah dan  kelompok lagu yang salah
