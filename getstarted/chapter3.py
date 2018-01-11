import tensorflow as tf

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W*x +b

#Konstanta diinisialisasi saat Anda memanggil tf.constant, dan nilainya tidak akan pernah berubah. Sebaliknya, variabel tidak diinisialisasi saat Anda memanggil tf.Variable. Untuk menginisialisasi semua variabel dalam program TensorFlow, Anda harus secara eksplisit memanggil operasi khusus sebagai berikut:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b,[1.0])

sess.run([fixW, fixb])

print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))


optimezer = tf.train.GradientDescentOptimizer(0.01)
train = optimezer.minimize(loss)

sess.run(init) #reset value yang salah

for i in range(1000):
	sess.run(train, {x: [1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))
