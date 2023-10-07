from collections import Counter

import matplotlib.pyplot as plt


def plot_losses(loss_values, val_losses):
    x0 = list(range(1, len(loss_values) + 1))

    # Plot loss of train and validation sets.
    plt.plot(x0, loss_values, label="Train loss")
    plt.plot(x0, val_losses, label="Validation loss")
    plt.title('Model loss')
    plt.legend(loc="upper right")
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
