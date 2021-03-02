# 训练及测试函数（加密，解密）
# from scipy.stats import ortho_group
import cupy as cp
import tensorflow as tf
import time
from numba import cuda 

def gen_ortho_matrix(dim):
    start = time.time()
    initializer = tf.keras.initializers.Orthogonal()
    ortho_matrix = initializer(shape=(dim, dim))
    end = time.time()
    ortho_matrix = ortho_matrix.numpy()

    print("Generated a {} orthogonal matrix in {} senconds.".format(dim, end-start))
    
    return ortho_matrix

def encryption_train(X,y):    
    X = cp.array(X.to_numpy())
    y = cp.array(y.to_numpy())
    
    # U1 is an orthogonal matrix
    U1 = cp.array(gen_ortho_matrix(X.shape[0]))
    
    # U2 is an invertible matrix
    if X.shape[1] > 1:
        U2 = cp.array(gen_ortho_matrix(X.shape[1]))
    else:
        U2 = cp.random.rand(1,1)
    
    X_enc = U1.dot(X).dot(U2)
    
    y_enc = U1.dot(y)
    return [X_enc,y_enc,U1,U2]

def decryption_train(X,y,U1,U2):
    X_dec = U1.T.dot(X).dot(cp.linalg.inv(U2))
    y_dec = U1.T.dot(y)
    return [X_dec,y_dec]

def encryption_test(X,U2):
    # U3 is an invertible matrix
    if X.shape[0] > 1:
        U3 = cp.array(gen_ortho_matrix(X.shape[0]))
    else:
        U3 = cp.random.rand(1,1)
    #from IPython import embed; embed()
    X_enc = U3.dot(X).dot(cp.linalg.inv(U2))
    return [X_enc,U3]

def decryption_test(y_enc,U3):
    return cp.linalg.inv(U3).dot(y_enc)

def estimator_OLS(X,y):
    return cp.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def predict(β̂,X):
    return X.dot(β̂)