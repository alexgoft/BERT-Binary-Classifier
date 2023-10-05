from collections import Counter

import matplotlib.pyplot as plt


def plot_losses(loss_values, val_losses,
                train_accuracies, val_accuracies,
                n_epochs):
    x0 = list(range(1, n_epochs + 1))

    _, axs = plt.subplots(1, 2, figsize=(10, 2))

    # Plot loss of train and validation sets.
    axs[0].plot(x0, loss_values, label="Train loss")
    axs[0].plot(x0, val_losses, label="Validation loss")
    axs[0].set_title('Model loss')
    axs[0].legend(loc="upper right")

    # # Plot accuracy of train and validation sets.
    # axs[1].plot(x0, train_accuracies, label="Train accuracy")
    # axs[1].plot(x0, val_accuracies, label="Validation accuracy")
    # axs[1].set_title('Model accuracy')
    # axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_column_histogram(df, column, title):
    classes = Counter(df[column])
    number_of_texts = list(classes.values())
    text_type = list(classes.keys())

    bars = plt.bar(text_type, number_of_texts)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0,
                 height, height, ha='center', va='bottom')
    plt.title(title)
    plt.xticks(rotation=20)
    plt.xlabel('Text type')
    plt.ylabel('Number of texts')
    plt.show()
