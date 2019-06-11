def plot_model_results(history):
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    val_acc = history.history['val_acc']
    train_acc = history.history['acc']

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [12, 5]

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    x_range = len(train_loss)
    
    # Loss Axis
    ax1.plot(list(range(x_range)), train_loss)
    ax1.plot(list(range(x_range)), val_loss)
    ax1.set_xlabel('Epoch')
    ax1.legend(['training loss', 'validation loss'])
    ax1.set_ylabel('Binary Cross Entropy Loss')

    # Accuracy Axis
    ax2.plot(list(range(x_range)), train_acc)
    ax2.plot(list(range(x_range)), val_acc)
    ax2.legend(['training accuracy', 'validation accuracy'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


def split_data(df, val_pct=.2, test_pct=.1):
    val_size   = int(val_pct * df.shape[0])
    test_size  = int(test_pct * df.shape[0])
    train_size = int((1.0 - (val_pct + test_pct)) * df.shape[0])
    df_train = df[:train_size]
    df_val = df[train_size : train_size + val_size]
    df_test = df[train_size + val_size : train_size + val_size + test_size]
    
    print(f"Training data length:   {len(df_train)}")
    print(f"Validation data length: {len(df_val)}")
    print(f"Test data length:       {len(df_test)}")

    y_train = df_train['RainTomorrow'].values
    y_test = df_test['RainTomorrow'].values
    y_val = df_val['RainTomorrow'].values
    X_train = df_train.drop(columns=['RainTomorrow']).values

    X_test = df_test.drop(columns=['RainTomorrow']).values
    X_val = df_val.drop(columns=['RainTomorrow']).values

    return X_train, X_val, X_test, y_train, y_val, y_test