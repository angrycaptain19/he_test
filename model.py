# vanilla model accuracy
import pandas as pd
import tensorflow as tf
import numpy  as np
from sklearn.model_selection import train_test_split
import math

def vanillaModel(x_data, y_data):

    # Feature columns describe how to use the input.
    # We are adding one numeric feature for each column of the training data
    my_feature_columns = []
    for key in x_data.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
        hidden_units=[9,9,3],
            # The model must choose between 3 classes.
        n_classes=2,
            ## We can also set the directory where model information will be saved.
        ##model_dir='models/iris'
        )

    def input_fn(features, labels, training=True, batch_size=256):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

    classifier.train(
        input_fn=lambda: input_fn(x_data, y_data, training=True),
        steps=1000)

    eval_result = classifier.evaluate(
            input_fn=lambda: input_fn(x_data, y_data), steps=1)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
    
def homomorphicEncryptionModel(X_enc, y_enc, x_data, H_enc):
    
    size = X_enc.shape[0]
    rng = np.random.default_rng()
    idx = np.arange(size)
    rng.shuffle(idx)

    train = math.floor(size * 0.7)
    idxTrain = np.sort(idx[:train])
    idxTest = np.sort(idx[train:])

    
    X_train = pd.DataFrame(X_enc.get()[idxTrain])
    X_train.columns = list(x_data.columns)
    
    y_train = pd.Series(y_enc.get()[idxTrain])

    X_test = pd.DataFrame(X_enc.get()[idxTest])
    X_test.columns = list(x_data.columns)
    
    y_test = pd.Series(y_enc.get()[idxTest])
    
    # y_enc.columns = list(y_data.columns)

    # Feature columns describe how to use the input.
    # We are adding one numeric feature for each column of the training data
    my_feature_columns = []
    for key in x_data.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    opti = tf.optimizers.Adam(learning_rate = 0.01)

    input_func= tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, 
                                                    y= y_train, 
                                                    batch_size=10, 
                                                    num_epochs=1000, 
                                                    shuffle=True)
    

    eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,
                                                          y=y_test, 
                                                          batch_size=10, 
                                                          num_epochs=1, 
                                                          shuffle=False)

    estimator = tf.estimator.DNNRegressor(hidden_units=[9,9,3], feature_columns=my_feature_columns, optimizer=opti, dropout=0.5)

    estimator.train(input_fn=input_func,steps=1000)
    
    result_eval = estimator.evaluate(input_fn=eval_input_func)

    predictions=[]
    for pred in estimator.predict(input_fn=eval_input_func):
        predictions.append(np.array(pred['predictions']).astype(float))
        
    from sklearn.metrics import mean_squared_error
    np.sqrt(mean_squared_error(y_test, predictions))**0.5
    
    accuracy=0
    for i in range(len(predictions)):
        if abs(predictions[i][0])-abs(y_test[i])<0.95:
            accuracy+=1
    print("HE accuracy: ", accuracy/float(len(y_test)))

def BenmarkModel(df):
    input_cols = list(df.columns)[:-1]
    output_cols = list(df.columns)[-1]
    X_train, X_test, y_train, y_test = train_test_split(df[input_cols], df[output_cols],test_size=0.3)

    # Feature columns describe how to use the input.
    # We are adding one numeric feature for each column of the training data
    my_feature_columns = []
    for key in X_test.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    opti = tf.optimizers.Adam(learning_rate = 0.01)

    input_func= tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, 
                                                    y= y_train, 
                                                    batch_size=10, 
                                                    num_epochs=1000, 
                                                    shuffle=True)
    

    eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,
                                                          y=y_test, 
                                                          batch_size=10, 
                                                          num_epochs=1, 
                                                          shuffle=False)

    estimator = tf.estimator.DNNRegressor(hidden_units=[9,9,3], feature_columns=my_feature_columns, optimizer=opti, dropout=0.5)

    estimator.train(input_fn=input_func,steps=1000)
    
    result_eval = estimator.evaluate(input_fn=eval_input_func)

    print(result_eval)

    predictions=[]
    for pred in estimator.predict(input_fn=eval_input_func):
        predictions.append(np.array(pred['predictions']).astype(float))
        
    from sklearn.metrics import mean_squared_error
    np.sqrt(mean_squared_error(y_test, predictions))**0.5

    y_test = np.array(y_test.values.tolist())

    accuracy=0
    for i in range(len(predictions)):
        if abs(predictions[i][0])-abs(y_test[i])<0.95:
            accuracy+=1
    print("Un-encrypted accuracy: ", accuracy/float(len(y_test)))