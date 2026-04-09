import numpy as np


class Attention:
    def __init__(self, hidden_size):
        fan_in = hidden_size
        self.W_h = np.random.randn(hidden_size, hidden_size) / np.sqrt(fan_in)
        self.W_s = np.random.randn(hidden_size, hidden_size) / np.sqrt(fan_in)
        self.v = np.random.randn(hidden_size, 1) / np.sqrt(fan_in)
        self.tanh = Tanh()

    def forward(self, H_enc, s_prev):
        # make sure shapes are column vectors
        s_prev = s_prev.reshape(-1, 1)
        scores = []
        for h_i in H_enc:
            h_i = h_i.reshape(-1, 1)
            z = self.W_h @ h_i + self.W_s @ s_prev
            self.tanh.forward(z)
            e_i = self.v.T @ self.tanh.output  # use stored output
            scores.append(e_i)
        scores = np.array(scores).reshape(-1)
        # softmax
        alpha = np.exp(scores - np.max(scores))
        alpha /= np.sum(alpha)

        # context vector
        c_t = np.sum(alpha[:, None] * H_enc, axis=0)

        self.cache = (H_enc, s_prev, alpha)

        return c_t, alpha


class LSTM:

    def __init__(self, n_neurons, input_dim=1):
        self.n_neurons = n_neurons
        self.input_dim = input_dim

        # weights for forget gate
        self.Uf = 0.1 * np.random.rand(n_neurons, input_dim)
        self.bf = 0.1 * np.random.rand(n_neurons, 1)
        self.Wf = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for input gate
        self.Ui = 0.1 * np.random.rand(n_neurons, input_dim)
        self.bi = 0.1 * np.random.rand(n_neurons, 1)
        self.Wi = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for output gate
        self.Uo = 0.1 * np.random.rand(n_neurons, input_dim)
        self.bo = 0.1 * np.random.rand(n_neurons, 1)
        self.Wo = 0.1 * np.random.rand(n_neurons, n_neurons)

        # weights for C_hat gate
        self.Ug = 0.1 * np.random.rand(n_neurons, input_dim)
        self.bg = 0.1 * np.random.rand(n_neurons, 1)
        self.Wg = 0.1 * np.random.rand(n_neurons, n_neurons)

        # gradients forget gate
        self.dUf = np.zeros((n_neurons, input_dim))
        self.dbf = np.zeros((n_neurons, 1))
        self.dWf = np.zeros((n_neurons, n_neurons))

        # gradients input gate
        self.dUi = np.zeros((n_neurons, input_dim))
        self.dbi = np.zeros((n_neurons, 1))
        self.dWi = np.zeros((n_neurons, n_neurons))

        # gradients output gate
        self.dUo = np.zeros((n_neurons, input_dim))
        self.dbo = np.zeros((n_neurons, 1))
        self.dWo = np.zeros((n_neurons, n_neurons))

        # gradients candidate gate
        self.dUg = np.zeros((n_neurons, input_dim))
        self.dbg = np.zeros((n_neurons, 1))
        self.dWg = np.zeros((n_neurons, n_neurons))

    def forward(self, X_t, h0=None, c0=None):
        T = X_t.shape[0]

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

        # sigmoids
        Sigmf = [Sigmoid() for t in range(T)]
        Sigmi = [Sigmoid() for t in range(T)]
        Sigmo = [Sigmoid() for t in range(T)]

        # Tanh
        Tanh1 = [Tanh() for t in range(T)]
        Tanh2 = [Tanh() for t in range(T)]

        # initial states (encoder or decoder)
        if h0 is not None:
            self.H[0] = h0
        if c0 is not None:
            self.C[0] = c0

        ht = self.H[0]
        ct = self.C[0]

        # calling the LSTM cell
        [H, C, self.Sigmf, self.Sigmi, self.Sigmo, self.Tanh1, self.Tanh2, F, O, I, C_tilde] \
            = self.LSTMCell(X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,
                            self.H, self.C, self.F, self.O, self.I, self.C_tilde)

        return self.H, self.C

    def LSTMCell(self, X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,
                 H, C, F, O, I, C_tilde):
        for t, xt in enumerate(X_t):
            xt = xt.reshape(self.input_dim, 1)

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

    def backward(self, dvalues):
        # dvalues is the derivative from the upper layer
        # we have two paths of derivation for calculation the dh_t-1
        T = self.T
        H = self.H
        C = self.C

        O = self.O
        I = self.I
        C_tilde = self.C_tilde
        F = self.F

        X_t = self.X_t

        Sigmf = self.Sigmf
        Sigmi = self.Sigmi
        Sigmo = self.Sigmo
        Tanh1 = self.Tanh1
        Tanh2 = self.Tanh2

        dht = np.zeros((self.n_neurons, 1))
        dct = np.zeros((self.n_neurons, 1))

        # actual BPTT

        for t in reversed(range(T)):
            xt = X_t[t].reshape(self.input_dim, 1)

            dht += dvalues[t].reshape(self.n_neurons, 1)

            Tanh2[t].backward(dht)
            dtanh2 = Tanh2[t].dinputs

            dct += np.multiply(O[t], dtanh2)

            dctdft = np.multiply(dct, C[t - 1])
            dctdit = np.multiply(dct, C_tilde[t])
            dctdct_tilde = np.multiply(dct, I[t])

            Tanh1[t].backward(dctdct_tilde)
            dtanh1 = Tanh1[t].dinputs

            Sigmf[t].backward(dctdft)
            dsigmf = Sigmf[t].dinputs

            Sigmi[t].backward(dctdit)
            dsigmi = Sigmi[t].dinputs

            Sigmo[t].backward(np.multiply(dht, Tanh2[t].output))
            dsigmo = Sigmo[t].dinputs

            dsigmfdUf = np.dot(dsigmf, xt.T)
            dsigmfdWf = np.dot(dsigmf, H[t - 1].T)

            self.dUf += dsigmfdUf
            self.dWf += dsigmfdWf
            self.dbf += dsigmf

            dsigmidUi = np.dot(dsigmi, xt.T)
            dsigmidWi = np.dot(dsigmi, H[t - 1].T)

            self.dUi += dsigmidUi
            self.dWi += dsigmidWi
            self.dbi += dsigmi

            dsigmodUo = np.dot(dsigmo, xt.T)
            dsigmodWo = np.dot(dsigmo, H[t - 1].T)

            self.dUo += dsigmodUo
            self.dWo += dsigmodWo
            self.dbo += dsigmo

            dtanh1dUg = np.dot(dtanh1, xt.T)
            dtanh1dWg = np.dot(dtanh1, H[t - 1].T)

            self.dUg += dtanh1dUg
            self.dWg += dtanh1dWg
            self.dbg += dtanh1

            dht = np.dot(self.Wf, dsigmf) + np.dot(self.Wi, dsigmi) + \
                  np.dot(self.Wo, dsigmo) + np.dot(self.Wg, dtanh1)

            dct = dct * F[t]


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


class Layer_Dense():

    def __init__(self, n_inputs, n_neurons):
        # note: we are using randn here in order to see if neg values are
        # clipped by the ReLU
        # import numpy as np
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # passing on the dot product as input for the next layer, as before
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs  # we're gonna need for backprop
        return self.output

    def backward(self, dvalues):
        # gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
