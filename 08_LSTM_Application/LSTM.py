import numpy as np


class LSTM:

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

        # forget gate
        self.Uf = 0.1 * np.random.rand(n_neurons, 1)
        self.bf = 0.1 * np.random.rand(n_neurons, 1)
        self.Wf = 0.1 * np.random.rand(n_neurons, n_neurons)

        # input gate
        self.Ui = 0.1 * np.random.rand(n_neurons, 1)
        self.bi = 0.1 * np.random.rand(n_neurons, 1)
        self.Wi = 0.1 * np.random.rand(n_neurons, n_neurons)

        # output gate
        self.Uo = 0.1 * np.random.rand(n_neurons, 1)
        self.bo = 0.1 * np.random.rand(n_neurons, 1)
        self.Wo = 0.1 * np.random.rand(n_neurons, n_neurons)

        # c tilde
        self.Ug = 0.1 * np.random.rand(n_neurons, 1)
        self.bg = 0.1 * np.random.rand(n_neurons, 1)
        self.Wg = 0.1 * np.random.rand(n_neurons, n_neurons)

    # def moh(self):
    #     pass

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons

        # weights for forget gate
        self.Uf = 0.1 * np.random.rand(n_neurons, 1)
        self.bf = 0.1 * np.random.rand(n_neurons, 1)
        self.Wf = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for input gate
        self.Ui = 0.1 * np.random.rand(n_neurons, 1)
        self.bi = 0.1 * np.random.rand(n_neurons, 1)
        self.Wi = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for output gate
        self.Uo = 0.1 * np.random.rand(n_neurons, 1)
        self.bo = 0.1 * np.random.rand(n_neurons, 1)
        self.Wo = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for C_hat gate
        self.Ug = 0.1 * np.random.rand(n_neurons, 1)
        self.bg = 0.1 * np.random.rand(n_neurons, 1)
        self.Wg = 0.1 * np.random.rand(n_neurons, n_neurons)

    def forward(self, X_t):
        T = max(X_t.shape)

        self.T = T
        self.X_t = X_t

        n_nuerons = self.n_neurons

        # gates
        self.H = [np.zeros((n_nuerons, 1)) for t in range(T + 1)]
        self.C = [np.zeros((n_nuerons, 1)) for t in range(T + 1)]
        self.C_tilde = [np.zeros((n_nuerons, 1)) for t in range(T)]

        self.F = [np.zeros((n_nuerons, 1)) for t in range(T)]
        self.O = [np.zeros((n_nuerons, 1)) for t in range(T)]
        self.I = [np.zeros((n_nuerons, 1)) for t in range(T)]

        ## derivates
        # derivates for input gate
        self.dUf = 0.1 * np.random.rand(n_nuerons, 1)
        self.dbf = 0.1 * np.random.rand(n_nuerons, 1)
        self.dWf = 0.1 * np.random.rand(n_nuerons, n_nuerons)

        # derivates for input gate
        self.dUi = 0.1 * np.random.rand(n_nuerons, 1)
        self.dbi = 0.1 * np.random.rand(n_nuerons, 1)
        self.dWi = 0.1 * np.random.rand(n_nuerons, n_nuerons)

        # derivates for output gate
        self.dUo = 0.1 * np.random.rand(n_nuerons, 1)
        self.dbo = 0.1 * np.random.rand(n_nuerons, 1)
        self.dWo = 0.1 * np.random.rand(n_nuerons, n_nuerons)

        # derivates for C_hat gate
        self.dUg = 0.1 * np.random.rand(n_nuerons, 1)
        self.dbg = 0.1 * np.random.rand(n_nuerons, 1)
        self.dWg = 0.1 * np.random.rand(n_nuerons, n_nuerons)

        # sigmoids
        Sigmf = [Sigmoid() for t in range(T)]
        Sigmi = [Sigmoid() for t in range(T)]
        Sigmo = [Sigmoid() for t in range(T)]

        # Tanh
        Tanh1 = [Tanh() for t in range(T)]
        Tanh2 = [Tanh() for t in range(T)]

        ht = self.H[0]
        ct = self.C[0]

        # calling the LSTM cell
        [H, C, self.Sigmf, self.Sigmi, self.Sigmo, self.Tanh1, self.Tanh2, F, O, I, C_tilde] \
            = self.LSTMCell(X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,
                            self.H, self.C, self.F, self.O, self.I, self.C_tilde)


    def LSTMCell(self, X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,
                 H, C, F, O, I, C_tilde):
        for t, xt in enumerate(X_t):
            xt = xt.reshape(1, 1)

            # forget gate
            outf = np.dot(self.Uf, xt) + np.dot(self.Wf, ht) + self.bf
            Sigmf[t].forward(outf)
            ft = Sigmf[t].output

            # input gate
            outi = np.dot(self.Ui, xt) + np.dot(self.Wi, ht) + self.bi
            Sigmi[t].forward(outi)
            it = Sigmi[t].output

            # output gate
            outo = np.dot(self.Uo, xt) + np.dot(self.Wo, ht) + self.bo
            Sigmo[t].forward(outo)
            ot = Sigmo[t].output

            # c tilde
            outct_tilde = np.dot(self.Ug, xt) + np.dot(self.Wg, ht) + self.bg
            Tanh1[t].forward(outct_tilde)
            ct_tilde = Tanh1[t].output

            ct = np.multiply(ft, ct) + np.multiply(it, ct_tilde)

            Tanh2[t].forward(ct)
            ht = np.multiply(Tanh2[t].output, ot)

            H[t + 1] = ht
            C[t + 1] = ct
            C_tilde[t] = ct_tilde

            F[t] = ft
            O[t] = ot
            I[t] = it

        return H, C, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2, F, O, I, C_tilde


class Sigmoid:

    def forward(self, M):
        # Sigmoid activation function
        # σ(x) = 1 / (1 + e^{-x})
        sigm = np.clip(1 / (1 + np.exp(-M)), 1e-7, 1 - 1e-7)

        # output = σ(M)
        self.output = sigm

        # store σ(M) because derivative uses it
        # σ'(x) = σ(x)(1 - σ(x))
        self.inputs = sigm

    def backward(self, dvalues):
        # dvalues = dL/dσ
        # sigm = σ(x)
        sigm = self.inputs

        # derivative of sigmoid
        # dσ/dx = σ(x)(1 - σ(x))
        deriv = np.multiply(sigm, (1 - sigm))

        # dL/dx = dL/dσ * dσ/dx
        self.dinputs = np.multiply(deriv, dvalues)


class Tanh:

    def forward(self, inputs):
        # a      = (Wh * h_t-1) + (Wx * X) + b
        # h      = tanh(a)
        # output = h, inputs = a
        # output = tanh(input)
        self.output = np.tanh(inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        # dvalues    = dL/dh
        # h          = output = tanh(a)
        # dh/da      = 1 - tanh(a)^2 = 1 - h^2
        deriv = 1 - self.output ** 2

        # dL/da = dL/dh     * dh/da
        #       = (y-y_hat) * (1-tanh^2)
        #       = dvalues   * deriv
        self.dinputs = np.multiply(deriv, dvalues)