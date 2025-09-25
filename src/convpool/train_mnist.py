# src/convpool/train_mnist.py
import os, argparse, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# CPU 병렬설정(원하면 숫자 조절)
tf.config.threading.set_intra_op_parallelism_threads(0)  # 0=자동
tf.config.threading.set_inter_op_parallelism_threads(0)

def build_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main(args):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]

    model = build_model()
    model.summary()

    os.makedirs("data", exist_ok=True)

    ckpt_path = "data/mnist_cnn.ckpt.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, save_weights_only=True
        ),
    ]

    t0 = time.time()
    hist = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )
    dur = time.time() - t0
    print(f"Training took {dur:.1f}s for {len(hist.history['loss'])} epochs.")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test acc: {test_acc:.4f}")

    # 체크포인트가 더 좋으면 그걸 불러와서 최종 저장
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    final_path = "data/mnist_cnn.weights.h5"
    model.save_weights(final_path)
    print(f"Saved weights -> {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)       # 먼저 10으로 테스트 권장
    parser.add_argument("--batch-size", type=int, default=128)  # CPU면 64~256 사이에서 맞춰보세요
    parser.add_argument("--patience", type=int, default=3)      # 개선 없으면 조기 종료
    args = parser.parse_args()
    main(args)
