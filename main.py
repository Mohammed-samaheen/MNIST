import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageDraw
from generator import Generator
from discriminator import Discriminator
import tensorflow_probability as tfp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


batch_size = 1
codings_size = 128

generator, discriminator = Generator(), Discriminator()
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.5)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
sample_array = tf.convert_to_tensor(np.array(np.meshgrid(np.arange(28), np.arange(28))).T.reshape(-1, 2))

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints', max_to_keep=3)

# @tf.function
def plot(points):
    img = Image.new('L', (28, 28))
    draw = ImageDraw.Draw(img)
    draw.line(points.tolist(), fill=255, width=2)

    return (np.array(img, dtype="float32").reshape(28, 28, 1) - 127.5) / 127.5

@tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
def tf_function(input):
    return tf.numpy_function(plot, [input], tf.float32)


@tf.function
def sample_trajectories(probability):
    probability_trajectories = tf.split(probability, 5)
    dist = tfp.distributions.Categorical(probs=probability_trajectories)
    trajectory_index = dist.sample()

    trajectory_probability = tf.gather(probability_trajectories, trajectory_index, axis=1, batch_dims=1)
    trajectory_loss = tf.math.reduce_sum(tf.math.log(trajectory_probability))
    trajectory = tf.gather(sample_array, trajectory_index, axis=0, batch_dims=1)

    return tf.reshape(trajectory, [-1]), tf.expand_dims(trajectory_loss, 0)


@tf.function
def generator_loss(d_out):
    return cross_entropy(tf.ones_like(d_out), d_out)


@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss)*-1


# @tf.function
def train_step(images):
    noise = tf.random.normal(shape=[batch_size, codings_size])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        probability_trajectories = generator(noise, training=True)

        trajectory_index, loss_list = tf.vectorized_map(sample_trajectories, probability_trajectories, False)
        fake_img = tf.vectorized_map(tf_function, trajectory_index)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(tf.reshape(fake_img,(batch_size, 28, 28, 1)), training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return g_loss, d_loss


gpus = tf.config.list_physical_devices('GPU')

(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
# Scale the pixel intensities down to the [0,1] range by dividing them by 255.0 
X_train = (X_train - 127.5) / 127.5

# Creating a Dataset to iterate through the images
train_filter = np.where((y_train == 1))
X_train, y_train = X_train[train_filter], y_train[train_filter]


dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)

dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

print("---------------------------------------------------------------")


def train_gan(dataset, n_epochs):
    for epoch in range(n_epochs):
        for batch_num, X_batch in dataset.enumerate():
            # phase 1 : training the discriminator

            # if batch_num % 100 == 0:


            g_loss, d_loss = train_step(X_batch)
            print(f"Epoch : {epoch}, g_loss:{g_loss}  d_loss:{d_loss}")
            if batch_num % 150 == 0:
                noise = tf.random.normal(shape=[1, codings_size])
                probability = generator.predict(noise)
                points = sample_trajectories(probability[0])[0].numpy()
                img = Image.new('L', (28, 28))
                draw = ImageDraw.Draw(img)
                draw.line(points.tolist(), fill=255, width=2)
                img.save(f"./output/img-{batch_num}.jpeg")
        if epoch % 15==0:
            manager.save()



n_epochs = 5000
checkpoint.restore(manager.latest_checkpoint)
train_gan(dataset, n_epochs)
