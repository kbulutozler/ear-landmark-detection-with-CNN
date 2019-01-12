from utilities import load_data



X, Y = load_data(size=10)
print(X.shape)
print(Y.shape)

X, Y = load_data(test=True, test_size=10)
print(X.shape)
print(Y.shape)


