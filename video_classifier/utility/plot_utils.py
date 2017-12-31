from matplotlib import pyplot as plt


def plot_history_2win(history):
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='g', label='Train')
    plt.plot(history.history['val_acc'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='g', label='Train')
    plt.plot(history.history['val_loss'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def create_history_plot(history, model_name):
    plt.title('Accuracy and Loss (' + model_name + ')')
    plt.plot(history.history['acc'], color='g', label='Train Accuracy')
    plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    plt.plot(history.history['loss'], color='r', label='Train Loss')
    plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()


def plot_history(history, model_name):
    create_history_plot(history, model_name)
    plt.show()


def plot_and_save_history(history, model_name, file_path):
    create_history_plot(history, model_name)
    plt.savefig(file_path)
