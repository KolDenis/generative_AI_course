import os
from tensorflow.keras import models, datasets
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = models.load_model(os.getcwd()+"/model.keras")
    print("Model loaded successfully.")
    return model

def load_data():
    (_, _), (X_test, y_test) = datasets.mnist.load_data()
    X_test = X_test / 255.0
    X_test = X_test.reshape(-1, 28 * 28)
    return X_test, y_test

def plot_class_metrics(predictions, y_test, num_classes=10):
    predicted_classes = np.argmax(predictions, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_class, predicted_class in zip(y_test, predicted_classes):
        cm[true_class, predicted_class] += 1

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar(label="Количество")

    plt.xticks(np.arange(num_classes), labels=np.arange(num_classes))
    plt.yticks(np.arange(num_classes), labels=np.arange(num_classes))
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица соответствий")

    # Подписи внутри ячеек
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

def plot_examples(X_examples, Y_examples):
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))

    axs[0, 0].imshow(X_examples[0].reshape(28, 28), cmap='viridis')
    axs[0, 0].set_title(Y_examples[0])

    axs[0, 1].imshow(X_examples[1].reshape(28, 28), cmap='viridis')
    axs[0, 1].set_title(Y_examples[1])

    axs[1, 0].imshow(X_examples[2].reshape(28, 28), cmap='viridis')
    axs[1, 0].set_title(Y_examples[2])

    axs[1, 1].imshow(X_examples[3].reshape(28, 28), cmap='viridis')
    axs[1, 1].set_title(Y_examples[3])

    plt.tight_layout()
    plt.show()

def main():
    model = load_model()
    X_test, y_test = load_data()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Overall Test Loss: {loss:.4f}")
    print(f"Overall Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(X_test)

    plot_class_metrics(predictions, y_test, num_classes=10)

    random_indices = np.random.choice(len(X_test), 4, replace=False)
    X_examples = X_test[random_indices]
    Y_examples = y_test[random_indices]
    plot_examples(X_examples, Y_examples)


if __name__ == '__main__':
    main()
