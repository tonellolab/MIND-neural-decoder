from __future__ import print_function, division

from keras.layers import Input, Dense, GaussianNoise, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical

import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt


def get_max_idx_loglikelihood(y,x):
    # maxL for AWGN channel
    N = np.size(x,0)
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = np.linalg.norm(y[0,:]-x[i,:])
    return np.argmin(distances)

def from_zero_mean_bits_to_digit(x):
    #convert to binary representation
    x = (x + 1) / 2
    N = np.size(x,0)
    d = np.size(x,1)
    digits = np.zeros((N,1))
    for i in range(d):
        digits[:,0] = digits[:,0] + (2**i)*x[:,d-i-1]
    return digits

def from_digit_to_zero_mean_bits(x,k):
    # convert from digit to zero mean
    M = len(x)
    output = np.zeros((M,k))
    for i in x:
        output[i,:] = np.transpose(np.fromstring(np.binary_repr(i, width=k),np.int8) - 48)
    output = 2*output-1
    return output


class MIND():
    def __init__(self, k, EbN0):

        # Input shape
        self.latent_dim = k
        self.EbN0 = EbN0

        # Noise std based on EbN0 in dB
        eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))
        self.eps = eps

        optimizer = Adam(0.01, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        y_x = Input(shape=(self.latent_dim,))
        y = Input(shape=(self.latent_dim,))

        # The discriminator takes as input joint or marginal vectors
        d_y_x = self.discriminator(y_x)
        d_y = self.discriminator(y)

        # Train the discriminator
        self.combined = Model([y_x, y], [d_y_x,d_y])

        # choose the loss function based on the f-divergence type
        self.combined.compile(loss=['binary_crossentropy','binary_crossentropy'],loss_weights=[1,1], optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(100, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GaussianNoise(0.3))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(2**self.latent_dim, activation='sigmoid'))

        model.summary()

        T = Input(shape=(self.latent_dim,))
        D = model(T)

        return Model(T, D)

    def train(self, epochs, batch_size=40):

        # Adversarial ground truths
        M = 2**self.latent_dim
        valid = np.ones((batch_size, M))
        h_y_x = np.zeros((batch_size, 1))
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Generate data to transmit
            data_tx = 2*np.random.randint(2, size=(batch_size, self.latent_dim))-1

            # Convert to digits for the supervised loss function
            digits = from_zero_mean_bits_to_digit(data_tx)

            eps = self.eps

            # Simulate the AWGN noise and produce the received samples
            data_rx = data_tx + eps*np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Sample from the marginal of the received samples
            data_y = 2*np.random.randint(2, size=(batch_size, self.latent_dim))-1 + eps*np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the discriminator alternating joint and marginal samples
            d_loss = self.combined.train_on_batch([data_rx,data_y],[1-to_categorical(digits, num_classes=M, dtype ="uint8"),valid])

            # Get the discriminator output (density ratios) and the densities
            D_value_1 = self.discriminator.predict(data_rx)
            R = (1 - D_value_1) / D_value_1

            # To avoid numerical errors, you may want to normalize the outputs
            L1_norm = np.expand_dims(np.sum(R,axis=1),axis=-1)*np.ones((1,np.size(R,axis=1)))
            R = R/L1_norm

            # Real-time estimate of source entropy, conditional entropy, mutual information, probability of error
            P_x_y = np.mean(R,axis=0) # a-posteriori estimate
            H_x = -P_x_y.dot(np.log2(P_x_y)) # batch source entropy estimate
            for i in range(batch_size):
                R[R==0]=1 # to avoid NaN
                h_y_x[i] = -R[i,:].dot(np.log2(R[i,:].T)) # instantaneous conditional entropy estimate

            H_y_x = np.mean(h_y_x,axis=0) # batch conditional entropy estimate
            MI = H_x-H_y_x # batch mutual information estimate
            P_error = 1-np.mean(np.max(R,axis=1),axis=0) # prob. of error

            # Plot the progress
            print ("%d [D total loss : %f, Batch source entropy : %f, B. cond. entropy: %f, B. MI: %f, B. prob. error: %f]" % (epoch, d_loss[0],H_x, H_y_x, MI, P_error))


    def test(self, test_size=1000):
        eps = self.eps
        BER = np.zeros((1, test_size))
        BER_maxL = np.zeros((1, test_size))
        h_y_x = np.zeros((test_size, 1))

        # Produce tx and rx samples
        data_tx = 2*np.random.randint(2, size=(test_size, self.latent_dim))-1
        noise = eps * np.random.normal(0, 1, (test_size, self.latent_dim))
        data_rx = data_tx + noise

        # Specify the alphabet for the MAP part
        alphabet = range(2**self.latent_dim)
        training_samples = from_digit_to_zero_mean_bits(alphabet,self.latent_dim)

        # Extract metrics for each transmitted sample
        for i in range(test_size):
            D_value_1 = self.discriminator.predict(np.expand_dims(data_rx[i,:],axis=0)) # density ratio
            R = (1 - D_value_1) / D_value_1 # a-posteriori estimates

            # To avoid numerical errors, you may want to normalize the outputs
            L1_single_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
            R = R/L1_single_norm

            h_y_x[i] = -R[0,:].dot(np.log2(R[0,:].T))  # instantaneous conditional entropy estimate
            max_idx = np.argmax(R) # MAP criterion
            max_idx_LL = get_max_idx_loglikelihood(np.expand_dims(data_rx[i,:],axis=0), training_samples) # maxL criterion

            logical_bits = training_samples[max_idx, :] == data_tx[i, :] # comparison in the MIND indices
            BER[0, i] = 1-sum(logical_bits)/self.latent_dim # instantaneous bit-error-rate with MIND

            logical_bits_LL = training_samples[max_idx_LL, :] == data_tx[i, :] # comparison in the maxL indices
            BER_maxL[0, i] = 1 - sum(logical_bits_LL) / self.latent_dim # instantaneous bit-error-rate with maxL

        D_all = self.discriminator.predict(data_rx) # get all density-ratios

        R_all = (1 - D_all)/D_all # get all a-posteriori probs

        # To avoid numerical errors, you may want to normalize the outputs
        L1_norm = np.expand_dims(np.sum(R_all, axis=1), axis=-1) * np.ones((1, np.size(R_all, axis=1)))
        R_all = R_all / L1_norm

        # Estimate of source entropy, conditional entropy, mutual information, probability of error
        P_x_y = np.mean(R_all, axis=0)  # a-posteriori estimate
        H_x = -P_x_y.dot(np.log2(P_x_y))  # source entropy estimate
        H_y_x = np.nanmean(h_y_x, axis=0)  # conditional entropy estimate
        MI = H_x - H_y_x  # mutual information estimate
        P_error = 1 - np.mean(np.max(R_all, axis=1), axis=0)  # prob. of error

        return np.sum(BER)/(test_size*self.latent_dim), np.sum(BER_maxL)/(test_size*self.latent_dim), H_x, H_y_x, MI, P_error, data_rx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=256)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=500)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=3000)
    args = parser.parse_args()

    test_size = int(args.test_size)


    k = 1
    M = 2**k # modulation order
    SNR_dB = range(-20,21)

    ber_total = np.zeros((len(SNR_dB),1))
    ber_total_LL = np.zeros((len(SNR_dB),1))
    H_x_total = np.zeros((len(SNR_dB),1))
    H_y_x_total = np.zeros((len(SNR_dB),1))
    MI_total = np.zeros((len(SNR_dB),1))
    P_error = np.zeros((len(SNR_dB),1))
    data_rx = np.zeros((len(SNR_dB),test_size,k))


    j = 0
    for SNR in SNR_dB:
        print(f'Actual SNR is:{SNR}')
        # Initialize dDIME
        mind = MIND(k, SNR)
        # Train
        mind.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        ber, ber_maxL, H_x, H_y_x, MI,P_e,y = mind.test(test_size=test_size)
        ber_total[j,0] = ber
        ber_total_LL[j,0] = ber_maxL
        H_x_total[j,0] = H_x
        H_y_x_total[j,0] = H_y_x
        MI_total[j,0] = MI
        P_error[j,0] = P_e
        data_rx[j,:,:] = y
        del mind
        j = j+1

    plt.figure(figsize=(6, 4), dpi=180)
    plt.plot(SNR_dB, ber_total, label='MIND')
    plt.plot(SNR_dB, ber_total_LL, label='MaxL')
    plt.plot(SNR_dB, P_error, label='P_error')
    plt.xlabel("Eb/N0")
    plt.ylabel("Prob. Error")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4), dpi=180)
    plt.plot(SNR_dB, H_x_total, label='Source Entropy')
    plt.plot(SNR_dB, H_y_x_total, label='Cond. Entropy')
    plt.plot(SNR_dB, MI_total, label='Mutual Information')
    plt.xlabel("Entropies")
    plt.ylabel("Eb/N0")
    plt.legend()
    plt.show()

    # Save on a .mat file the variables for further processing
    sio.savemat('MAP_MIND.mat', {'SNR': SNR_dB, 'BER': ber_total, 'BER_LL': ber_total_LL, 'source_entropy': H_x_total,'cond_entropy': H_y_x_total,'MI': MI_total, 'P_error': P_error,'data_rx': data_rx})
