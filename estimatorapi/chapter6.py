# NumPy sering digunakan untuk memuat, memanipulasi dan data preprocess
import numpy as np
import tensorflow as tf

# Deklarasikan daftar fitur. Kami hanya memiliki satu fitur numerik. Ada banyak
# jenis kolom lainnya yang lebih rumit dan berguna.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Pengukur adalah ujung depan untuk memanggil latihan (pas) dan evaluasi
# (inferensi). Ada banyak jenis standar seperti regresi linier,
#Klasifikasi linier, dan banyak pengklasifikasi dan regresi jaringan syaraf.
# Kode berikut memberikan estimator yang melakukan regresi linier.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow menyediakan banyak metode untuk membaca dan menyiapkan kumpulan data.
# Di sini kita menggunakan dua kumpulan data: satu untuk pelatihan dan satu untuk evaluasi
# Kita harus memberitahukan fungsinya berapa banyak batch
# data (num_epochs) yang kita inginkan dan seberapa besar setiap batch seharusnya.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Kita bisa memanggil 1000 langkah pelatihan dengan cara memanggil metode dan melewati
# data training set.
estimator.train(input_fn=input_fn, steps=1000)

# Di sini kita mengevaluasi seberapa baik model kita.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)