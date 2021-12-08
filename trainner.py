import tensorflow as tf
import tensorflow.keras as keras
from CBAM_ResNet import ResNet
from DataLoader import Loader, configure_for_performance
from Setting import setting_option


class Trainner(setting_option):
    def __init__(self):
        super(Trainner, self).__init__()

        self.model = ResNet()(self.input_shape, self.depth, self.num_class)

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_mean = keras.metrics.Mean()

        self.train_acc_metric = keras.metrics.CategoricalAccuracy()
        self.test_acc_metric = keras.metrics.CategoricalAccuracy()

        self.train_ds = Loader(self.train_root, self.class_list).load()
        self.test_ds = Loader(self.val_root, self.class_list).load()
        self.val_ds = Loader(self.test_root, self.class_list).load()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_acc_metric.update_state(y, logits)
        self.loss_mean.update_state(loss)

    @tf.function
    def test_step(self, x, y):
        logits = self.model(x, training=False)
        self.test_acc_metric.update_state(y, logits)

    def training(self):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.ckp_path, max_to_keep=None)

        train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        full_train_steps = self.traincnt // self.batchSZ
        full_val_steps = self.valcnt // self.batchSZ

        for epoch in range(self.epochs):
            print(f'Epoch ({epoch}/{self.epochs})')

            train_ds_shuffle = configure_for_performance(self.train_ds, self.traincnt, shuffle=True)
            train_ds_it = iter(train_ds_shuffle)

            val_ds_shuffle = configure_for_performance(self.val_ds, self.valcnt, shuffle=False)
            val_ds_it = iter(val_ds_shuffle)

            for step in range(full_train_steps):
                img, label = next(train_ds_it)
                loss = self.train_step(img, label)

                if step % 200 == 0:
                    print(f'step ({step}/{full_train_steps})  Loss : {loss}')

            result_loss = self.loss_mean.result()
            self.loss_mean.reset_states()

            train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_states()

            manager.save()

            print(f'Train [ Epoch ({epoch}/{self.epochs})   Loss : {result_loss}.4f   Acc : {train_acc}.4f ]')

            for step in range(full_val_steps):
                img, label = next(val_ds_it)
                self.test_step(img, label)

            val_acc = self.test_acc_metric.result()
            self.test_acc_metric.reset_states()

            print(f'Validation [ Epoch ({epoch}/{self.epochs})  Acc : {val_acc}.4f ]')

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', result_loss, step=epoch)
                tf.summary.scalar('Accuracy', train_acc, step=epoch)
                tf.summary.scalar('validation_Accuracy', val_acc, step=epoch)

    def testing(self):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)

        full_test_steps = self.testcnt // self.batchSZ
        test_ds_shuffle = configure_for_performance(self.test_ds, self.testcnt, shuffle=True)
        test_ds_it = iter(test_ds_shuffle)

        for epoch in range(self.epochs):
            ckpt_path = f'path_/{epoch+1}'
            ckpt.restore(ckpt_path)

            print(f'Epoch ({epoch}/{self.epochs})')

            for step in range(full_test_steps):
                img, label = next(test_ds_it)
                self.test_step(img, label)

            test_acc = self.test_acc_metric.result()
            self.test_acc_metric.reset_states()

            print(f'Validation [ Epoch ({epoch}/{self.epochs})  Acc : {test_acc}.4f ]')