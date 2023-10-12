from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


def reset_plot():
    """ Reset the plot."""
    plt.clf()
    plt.cla()
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """ Plot the ROC curve."""
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig(f'{output_dir}/roc_curve.png')
    reset_plot()


def plot_losses(loss_values, val_losses, output_dir):
    """ Plot the loss of the train and validation sets."""
    x0 = list(range(1, len(loss_values) + 1))

    # Plot loss of train and validation sets.
    plt.plot(x0, loss_values, label="Train loss")
    plt.plot(x0, val_losses, label="Validation loss")
    plt.title('Model loss')
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{output_dir}/train_val_loss.png')
    reset_plot()


def plot_column_histogram(df, column, title, output_dir_path,
                          # TODO: Generalize this.
                          x_label='Text type', y_label='Number of texts'):
    """ Plot the histogram of a column in a dataframe."""
    classes = Counter(df[column])
    number_of_texts = list(classes.values())
    text_type = list(classes.keys())

    bars = plt.bar(text_type, number_of_texts)
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0,
                 height, height, ha='center', va='bottom')
    plt.title(title + f' (Total: {len(df)})')
    plt.xticks(list(range(len(classes))), rotation=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    title = title.replace(' ', '_')
    plt.savefig(f'{output_dir_path}/{title}.png')
    reset_plot()


def plot_confusion_matrix(cm, output_dir):
    """ Plot the confusion matrix."""
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Not-News', 'News'])
    ax.yaxis.set_ticklabels(['Not-News', 'News'])
    # plt.show()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    reset_plot()
