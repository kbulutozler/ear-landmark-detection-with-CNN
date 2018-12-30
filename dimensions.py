from utilities import load_data

X, Y = load_data()
print(X.shape)
print(Y.shape)

X, Y = load_data(test=True)
print(X.shape)
print(Y.shape)


