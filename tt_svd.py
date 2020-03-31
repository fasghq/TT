def tt_svd(A, eps):
    d = len(A.shape)
    N = np.size(A)
    n = A.shape

    C = A # tmp tensor

    G = [] # tt-cores
    r = [] # tt-ranks
    r.append(1)

    for k in range(1, d):
        C = np.reshape(C, (r[k-1] * n[k-1], int(N / (r[k-1] * n[k-1]))))
  
        # calc low-rank approximation
        u, s, v = la.svd(C)
        sum = 0 
        nsize = np.size(s)
        rres = np.size(s)
        for rk in range(0, nsize):
            for m in range(rk+1, nsize):
                sum = sum + (s[m] ** 2)
            if (sum <= (eps ** 2) * la.norm(A)) and (rres > rk):
                rres = rk + 1 
            sum = 0
        r .append(rres) 

        G.append(np.reshape(u[:, :r[k]], (r[k-1], n[k-1], r[k])))
        s = np.diag(s)
        C = np.dot(s[:r[k], :r[k]], v[:r[k], :])
        N = (N * r[k]) / (n[k-1] * r[k-1])
    
    C = np.reshape(C, (C.shape[0], C.shape[1], 1))
    G.append(C) 

    return G, r