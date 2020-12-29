from mid_interface import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn
import time
from sklearn.decomposition import PCA


learning_rate = 0.001
batch_size = 5

def eval_conf_matrix(labels, predictions, params):
    num_classes = params['num_classes']
    conf_matrix = tf.confusion_matrix(labels, predictions, num_classes=num_classes)
    conf_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                  name="confusion_matrix",
                                  trainable=False,
                                  collections=[tf.GraphKeys.LOCAL_VARIABLES])

    update_op = tf.assign_add(conf_matrix_sum, conf_matrix)
    return tf.convert_to_tensor(conf_matrix_sum), update_op


# This is the part where we define the model
def model_fn(features, labels, mode, params):
    global_step = tf.train.get_global_step()

    # The input layer
    input_layer = tf.reshape(features, [-1, features.shape[-1]])

    # First dense layer
    dense = tf.layers.dense(inputs=input_layer,
                            units=params['dense_layers'][0]['units'],
                            activation=tf.nn.relu)

    # dropout applied to the dense layers
    dropout = tf.layers.dropout(inputs=dense, rate=params['dense_layers'][0]['dropout'])

    for dense_layer in params['dense_layers'][1:]:
        # interior dense layers
        dense = tf.layers.dense(inputs=dropout,
                                units=dense_layer['units'],
                                activation=tf.nn.relu)

        # If dropout is not necessary, set the rate to zero in params dictionary
        dropout = tf.layers.dropout(inputs=dense, rate=dense_layer['dropout'])

    # The logits will be calculated from the last dropout layer
    logits = tf.layers.dense(inputs=dropout, units=params['num_classes'])

    prediction_labels = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits)

    predictions = {
        'prediction_labels': prediction_labels,
        'probabilities': probabilities
    }

    #   This part is for the prediction part of the model
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    accuracy = tf.metrics.accuracy(labels, prediction_labels)
    confusion_matrix_tuple = eval_conf_matrix(labels, prediction_labels, params)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    # This is the evaluation part of the model
    if mode == tf.estimator.ModeKeys.EVAL:
        tf.summary.scalar("Evaluation_accuracy", accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops={'eval_accuracy': accuracy,
                             'confusion_matrix': confusion_matrix_tuple},
            predictions=predictions,
            evaluation_hooks=None
        )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

    train_hook_list = []
    train_tensor_logs = {
        'accuracy': accuracy[1],
        'loss': cross_entropy,
        'global_step': global_step
    }

    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensor_logs, every_n_iter=100
    ))

    #   This is the training part of the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("Training_accuracy", accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            train_op=train_op,
            eval_metric_ops=None,
            training_hooks=train_hook_list
        )


def STL_classifier(_):
    num_classes = 16
    x_train = read_data(path="data/HighDim.txt", labels=False)
    y_train = np.zeros((1024, 1))

    pca = PCA(n_components=3)
    x_train = pca.fit_transform(x_train)

    length = int(len(x_train) / num_classes)
    for i in range(num_classes):
        y_train[i * length:(i + 1) * length, 0] += i

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        num_epochs=7,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    saving_configuration = tf.estimator.RunConfig(keep_checkpoint_max=2)

    image_classifier = tf.estimator.Estimator(
        model_dir="model/2.2(3)",
        model_fn=model_fn,
        config=saving_configuration,
        params={
            'num_classes': num_classes,
            'dense_layers': [{'units': 512, 'dropout': 0.0}
                            ,{'units': 512, 'dropout': 0.0}
                            ,{'units': 512, 'dropout': 0.2}
                            ,{'units': 512, 'dropout': 0.2}]
        })

    start = time.time()

    for i in range(2):
        image_classifier.train(input_fn=train_input_fn, steps=None)
        metrices = image_classifier.evaluate(input_fn=eval_input_fn)
        print("\n\n******************************************************\nConfusion matrix for evaluation data:\n\n"
              + str(metrices['confusion_matrix']) + "\n\n\nOther evaluation matrices:"
              + "\nEvaluation accuracy:%.2f\tLoss:%.2f\tGlobal step:%d" % (
              metrices['eval_accuracy'] * 100, metrices['loss'], metrices['global_step'])
              + "\n******************************************************")

    end = time.time()
    print("******************************************************\nTraining time:%.4s seconds" % str(end - start)
          + "\n******************************************************\n\n")

    predictions = image_classifier.predict(
        input_fn=test_input_fn,
        yield_single_examples=False
    )

    accuracy = 0
    i = 0
    conf_matrix = np.zeros((16, 16))
    for epoch_result in predictions:
        accuracy += sklearn.metrics.accuracy_score(y_test[i * batch_size: (i + 1) * batch_size],
                                                   epoch_result['prediction_labels'])

        conf_matrix += sklearn.metrics.confusion_matrix(y_test[i * batch_size: (i + 1) * batch_size],
                                                        epoch_result['prediction_labels'], labels=range(16))

        i = i + 1
    print("\n******************************************************\nConfusion matrix for test data:\n\n" + str(
        conf_matrix)
          + "\n******************************************************")

    print(conf_matrix)

    print("\n******************************************************\nAccuracy for test data: %.2f"
          "\n******************************************************\n" % (accuracy / i * 100))


if __name__ == '__main__':
    tf.app.run(STL_classifier)