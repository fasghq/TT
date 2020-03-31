def tt_to_tensor(G, shape):
    B = G[0]
    for i in range(1, len(G)):
        X = np.reshape(G[i], (G[i].shape[0], G[i].shape[1] * G[i].shape[2]))
        B = np.reshape(B, (int(np.size(B) / G[i].shape[0]), G[i].shape[0]))
        B = np.dot(B, X)
    B = np.reshape(B, shape)
    return B