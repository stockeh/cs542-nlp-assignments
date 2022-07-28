# Nov 24, 2021
import copy
import numpy as np
import random
import torch
import time

import torch.nn.functional as F
import torch_optimizer as optim


class EarlyStopping(object):
    '''
    MIT License, Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
    es = EarlyStopping(patience=5)

    for epoch in range(n_epochs):
        # train the model for one epoch, on training set
        train_one_epoch(model, data_loader)
        # evalution on dev set (i.e., holdout from training)
        metric = eval(model, data_loader_dev)
        if es.step(metric):
            break  # early stop criterion is met, we can stop now
    '''

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


class TextNeuralNetwork():

    class Network(torch.nn.Module):
        def __init__(self, embed_args, n_hiddens_list, n_outputs, padding_idx,
                     activation_f='tanh', **kwargs):
            super().__init__()
            assert isinstance(embed_args, dict), 'embed_args must be a dict.'

            # args
            self.n_hiddens_list = n_hiddens_list
            self.n_outputs = n_outputs
            self.padding_idx = padding_idx

            if not isinstance(n_hiddens_list, list):
                raise Exception(
                    'Network: n_hiddens_list must be a list.')

            if len(n_hiddens_list) == 0 or n_hiddens_list[0] == 0:
                self.n_hidden_layers = 0
            else:
                self.n_hidden_layers = len(n_hiddens_list)

            # network varaibles
            self.n_hiddens_list = n_hiddens_list
            self.n_outputs = n_outputs

            activations = [
                torch.nn.Tanh,
                torch.nn.Sigmoid,
                torch.nn.ReLU,
                torch.nn.ELU,
                torch.nn.PReLU,
                torch.nn.ReLU6,
                torch.nn.LeakyReLU,
                torch.nn.Mish,
            ]
            names = [str(o.__name__).lower() for o in activations]
            try:
                activation = activations[names.index(
                    str(activation_f).lower())]
            except:
                raise NotImplementedError(
                    f'__init__: {activation_f=} is not yet implemented.')

            # build nnet
            self.model = torch.nn.ModuleList()

            if 'embeddings' in embed_args:
                # embed_args = {'embeddings': embeddings}
                embedding = torch.nn.Embedding.from_pretrained(
                    torch.from_numpy(embed_args['embeddings']).float(),
                    freeze=True, padding_idx=padding_idx)  # !!!
            else:
                # embed_args = {'num_embeddings': num_embeddings, 'embedding_dim', embedding_dim}
                assert embed_args.keys() & {
                    'num_embeddings', 'embedding_dim'}, 'embed_args must contain `num_embeddings` \
                        and `embedding_dim`'
                embedding = torch.nn.Embedding(embed_args['num_embeddings'], embed_args['embedding_dim'],
                                               padding_idx=padding_idx)
            self.num_embeddings = embedding.num_embeddings
            self.embedding_dim = embedding.embedding_dim
            self.model.add_module(f'embedding', embedding)  # !!!

            if 'lstm_args' in kwargs:
                # lstm_args = {'lstm_hidden_dim': 64, 'n_lstm_layers': 1}
                self.modeltype = 0  # int for fast check
                self.lstm_hidden_dim = kwargs['lstm_args'].get(
                    'lstm_hidden_dim', 64)
                self.n_lstm_layers = kwargs['lstm_args'].get(
                    'n_lstm_layers', 1)
                self.model.add_module(f'lstm', torch.nn.LSTM(
                    input_size=self.embedding_dim, hidden_size=self.lstm_hidden_dim,
                    num_layers=self.n_lstm_layers, batch_first=True))  # !!!
                ni = self.lstm_hidden_dim
            elif 'cnn_args' in kwargs:
                # cnn_args = [{'n_units': n_units, 'window': window}, ...]
                self.modeltype = 1
                convs = torch.nn.ModuleList()
                self.cnn_args = kwargs['cnn_args']
                l = 0
                nc = 1
                for conv_layer in self.cnn_args:
                    convs.add_module(f'conv_{l}', torch.nn.Conv2d(nc, conv_layer['n_units'], (
                        conv_layer['window'], self.embedding_dim), stride=1,
                        padding='same', padding_mode='zeros'))
                    convs.add_module(f'activation_{l}', activation())
                    convs.add_module(f'maxpool_{l}', torch.nn.MaxPool2d(
                        (2, 1), stride=(2, 1)))
                    l += 1
                    nc = conv_layer['n_units']
                ni = nc * self.embedding_dim
                self.model.append(convs)  # !!!
                self.model.add_module('flatten', torch.nn.Flatten())  # !!!
            else:
                # do not specify `lstm_args` or `cnn_args`
                self.modeltype = 2
                ni = self.embedding_dim

            self.model.add_module(f'dropout', torch.nn.Dropout(0.2))  # !!!
            l = 0
            # add fully-connected layers
            if self.n_hidden_layers > 0:
                for i, n_units in enumerate(n_hiddens_list):
                    self.model.add_module(
                        f'linear_{l}', torch.nn.Linear(ni, n_units))  # !!!
                    self.model.add_module(
                        f'activation_{l}', activation())  # !!!
                    ni = n_units
                    l += 1
            self.model.add_module(
                f'output_{l}', torch.nn.Linear(ni, n_outputs))

            # self.model.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        def _init_lstm_hc(self, X):
            h = torch.zeros(
                (self.n_lstm_layers, X.size(0), self.lstm_hidden_dim))
            c = torch.zeros(
                (self.n_lstm_layers, X.size(0), self.lstm_hidden_dim))
            torch.nn.init.xavier_normal_(h)
            torch.nn.init.xavier_normal_(c)
            if next(self.model.lstm.parameters()).is_cuda:
                h = h.to('cuda')
                c = c.to('cuda')
            return h, c

        def forward_all_outputs(self, X, lens):
            Ys = []
            # embedding (bs, embedding_dim)
            Ys.append(self.model.embedding(X))
            if self.modeltype == 0:
                # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    Ys[-1], lens, batch_first=True, enforce_sorted=False)
                lstm_hc = self._init_lstm_hc(X)  # (h,c)
                out, (hs, cs) = self.model.lstm(packed, lstm_hc)
                # lstm, (bs, embedding_dim)
                Ys.append(hs[-1])
                s = 2
            elif self.modeltype == 1:
                Ys[-1] = torch.unsqueeze(Ys[-1], 1)
                for layer in self.model[1]:
                    Ys.append(layer(Ys[-1]))
                # mean over timesteps (BS, C, T, E) -> (BS, C, E)
                Ys.append(torch.mean(Ys[-1], dim=2))
                s = 2
            elif self.modeltype == 2:
                # mean over timesteps (BS, T, E) -> (BS, E)
                Ys.append(torch.mean(Ys[-1], dim=1))
                s = 1
            else:
                raise NotImplementedError(
                    f'Forward pass not implemented for {self.modeltype=}')
            for layer in self.model[s:]:
                Ys.append(layer(Ys[-1]))

            return Ys

        def forward(self, X, lens):
            Ys = self.forward_all_outputs(X, lens)
            return Ys[-1]

    def __init__(self, embed_args, n_hiddens_list, n_outputs, padding_idx,
                 activation_f='tanh', use_gpu=True, seed=None, **kwargs):
        """
        # embed_args = {'embeddings': embeddings}
        embed_args = {'num_embeddings': len(vocab), 'embedding_dim': 100}
        lstm_args = {'lstm_hidden_dim': 64, 'n_lstm_layers': 1}
        # cnn_args = [{'n_units': 5, 'window': 1}, {'n_units': 5, 'window': 2},
        #             {'n_units': 5, 'window': 3}, {'n_units': 5, 'window': 5}]
        nnet = tnn.TextNeuralNetworkClassifier(embed_args, n_hiddens_list=[0], n_outputs=len(np.unique(Ttrain)),
                                            lstm_args=lstm_args, # cnn_args=cnn_args, or lstm_args=lstm_args,
                                            padding_idx=wtoi['<pad>'], activation_f='relu', 
                                            use_gpu=True, seed=1234)
        """
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
        self.seed = seed

        if use_gpu and not torch.cuda.is_available():
            print('\nGPU is not available. Running on CPU.\n')
            use_gpu = False
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        # Build nnet
        self.model = self.Network(embed_args, n_hiddens_list, n_outputs, padding_idx,
                                  activation_f, **kwargs)
        self.model.to(self.device)
        self.loss = None
        self.optimizer = None

        # Member variables for standardization
        self.Tmeans = None
        self.Tstds = None

        # Bookkeeping
        self.train_error_trace = []
        self.val_error_trace = []
        self.n_epochs = None
        self.batch_size = None
        self.training_time = None

    def __repr__(self):
        str = f'{type(self).__name__}({self.model.num_embeddings=}, {self.model.embedding_dim=}, {self.model.n_outputs=},'
        str += f' {self.use_gpu=}, {self.seed=})'
        if self.training_time is not None:
            str += f'\n   Network was trained for {self.n_epochs} epochs'
            str += f' that took {self.training_time:.4f} seconds.\n   Final objective values...'
            str += f' train: {self.train_error_trace[-1]:.3f},'
            if len(self.val_error_trace):
                str += f'val: {self.val_error_trace[-1]:.3f}'
        else:
            str += '  Network is not trained.'
        return str

    def summary(self):
        print(self.model)

    def _standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result

    def _unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans

    def _setup_standardize(self, T):
        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            self.Tconstant = self.Tstds == 0
            self.TstdsFixed = copy.copy(self.Tstds)
            self.TstdsFixed[self.Tconstant] = 1

    def _padbatch(self, X):
        maxlen = max(map(len, X))
        pads, lens = [], []
        for x in X:
            pads.append(x + [self.model.padding_idx]*(maxlen-len(x)))
            lens.append(len(x))
        return torch.LongTensor(pads).to(self.device), lens

    def _train(self, training_data, validation_data):
        # training
        #---------------------------------------------------------------#
        Xtrain, Ttrain = training_data
        running_loss = 0
        self.model.train()

        for i in range(0, len(Xtrain), self.batch_size):
            # should we .to(self.device) here?
            X, lens = self._padbatch(Xtrain[i:i+self.batch_size])
            T = torch.LongTensor(
                Ttrain[i:i+self.batch_size]).flatten().to(self.device)
            # compute prediction error
            Y = self.model(X, lens)
            error = self.loss(Y, T)

            # backpropagation
            self.optimizer.zero_grad()
            error.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # unaveraged sum of losses over all samples
            # https://discuss.pytorch.org/t/interpreting-loss-value/17665/10
            running_loss += error.item() * len(X)

        # maintain loss over every epoch
        self.train_error_trace.append(running_loss / len(Xtrain))

        # validation
        #---------------------------------------------------------------#
        if validation_data is not None:
            Xval, Tval = validation_data
            running_loss = 0
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(Xval), self.batch_size):
                    X, lens = self._padbatch(Xval[i:i+self.batch_size])
                    T = torch.LongTensor(
                        Tval[i:i+self.batch_size]).flatten().to(self.device)
                    Y = self.model(X, lens)
                    error = self.loss(Y, T)
                    running_loss += error.item() * len(X)

                self.val_error_trace.append(running_loss / len(Xval))

    def train(self, Xtrain, Ttrain, n_epochs, batch_size, learning_rate,
              opt='adam', weight_decay=0, early_stopping=False,
              validation_data=None, shuffle=False, verbose=True):

        self._setup_standardize(Ttrain)  # only occurs once

        if validation_data is not None:
            assert len(
                validation_data) == 2, 'validation_data: must be (Xval, Tval).'
            Xval, Tval = validation_data[0], validation_data[1]

        if not isinstance(self.loss, (torch.nn.NLLLoss, torch.nn.CrossEntropyLoss)):
            Ttrain = self._standardizeT(Ttrain)
            if validation_data is not None:
                Tval = self._standardizeT(Tval)

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if self.loss is None:
            self.loss = torch.nn.MSELoss()

        if self.optimizer is None:
            optimizers = [
                torch.optim.SGD,
                torch.optim.Adam,
                optim.A2GradExp,
                optim.A2GradInc,
                optim.A2GradUni,
                optim.AccSGD,
                optim.AdaBelief,
                optim.AdaBound,
                optim.AdaMod,
                optim.Adafactor,
                optim.AdamP,
                optim.AggMo,
                optim.Apollo,
                optim.DiffGrad,
                optim.Lamb,
                optim.NovoGrad,
                optim.PID,
                optim.QHAdam,
                optim.QHM,
                optim.RAdam,
                optim.Ranger,
                optim.RangerQH,
                optim.RangerVA,
                optim.SGDP,
                optim.SGDW,
                optim.SWATS,
                optim.Yogi,
            ]
            names = [str(o.__name__).lower() for o in optimizers]
            try:
                self.optimizer = optimizers[names.index(str(opt).lower())](
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            except:
                raise NotImplementedError(
                    f'train: {opt=} is not yet implemented.')

        print_every = n_epochs // 10 if n_epochs > 9 else 1
        if early_stopping:
            es = EarlyStopping(patience=10)
        # training loop
        #---------------------------------------------------------------#
        start_time = time.time()
        for epoch in range(n_epochs):
            if shuffle:  # shuffle after every epoch
                if self.seed is not None:
                    random.seed(self.seed + epoch)
                c = list(zip(Xtrain, Ttrain))
                random.shuffle(c)
                Xtrain, Ttrain = zip(*c)
                Ttrain = np.vstack(Ttrain)
                if validation_data is not None:
                    c = list(zip(Xval, Tval))
                    random.shuffle(c)
                    Xval, Tval = zip(*c)
                    Tval = np.vstack(Tval)
            # forward, grad, backprop
            self._train((Xtrain, Ttrain), (Xval, Tval)
                        if validation_data is not None else None)
            if early_stopping and validation_data is not None and es.step(self.val_error_trace[-1]):
                self.n_epochs = epoch + 1
                break  # early stop criterion is met, we can stop now
            if verbose and (epoch + 1) % print_every == 0:
                st = f'Epoch {epoch + 1} error - train: {self.train_error_trace[-1]:.5f},'
                if validation_data is not None:
                    st += f' val: {self.val_error_trace[-1]:.5f}'
                print(st)
        self.training_time = time.time() - start_time

        # remove data from gpu, needed?
        torch.cuda.empty_cache()

        # convert loss to likelihood
        # TODO: append values to continue with training
        if isinstance(self.loss, (torch.nn.NLLLoss, torch.nn.CrossEntropyLoss)):
            self.train_error_trace = np.exp(
                -np.asarray(self.train_error_trace))
            if validation_data is not None:
                self.val_error_trace = np.exp(
                    -np.asarray(self.val_error_trace))

        return self

    def use(self, X, all_output=False, detach=True):
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                X, lens = self._padbatch(X)
                Ys = self.model.forward_all_outputs(X, lens)
                Ys[-1] = self._unstandardizeT(Ys[-1])
                if detach:
                    X = X.detach().cpu().numpy()  # is this needed?
                    Ys = [Y.detach().cpu().numpy() for Y in Ys]
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        return Ys if all_output else Ys[-1]


class TextNeuralNetworkClassifier(TextNeuralNetwork):

    class Network(TextNeuralNetwork.Network):
        def __init__(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)
            # not needed if CrossEntropyLoss is used.
            # self.model.add_module(f'log_softmax', torch.nn.LogSoftmax(dim=1))

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # TODO: only supporting CrossEntropyLoss as use function now computes softmax
        # self.loss = torch.nn.NLLLoss()
        # CrossEntropyLoss computes LogSoftmax then NLLLoss
        self.loss = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)

    def use(self, X, all_output=False, detach=True):
        """
        Return:
            if all_output: predicted classes, all layers + softmax
            else: predicted classes
        """
        # turn off gradients and other aspects of training
        self.model.eval()
        try:
            with torch.no_grad():
                X, lens = self._padbatch(X)
                Ys = self.model.forward_all_outputs(X, lens)
                # probabilities
                Ys.append(F.softmax(Ys[-1], dim=1))
                if detach:
                    X = X.detach().cpu().numpy()  # is this needed?
                    Ys = [Y.detach().cpu().numpy() for Y in Ys]
        except RuntimeError:
            raise
        finally:
            torch.cuda.empty_cache()
        Y = Ys[-1].argmax(1).reshape(-1, 1)
        return (Y, Ys) if all_output else Y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.switch_backend('tkagg')

    def rmse(A, B): return np.sqrt(np.mean((A - B)**2))
    def accuracy(A, B): return 100. * np.mean(A == B)
    br = ''.join(['-']*8)

    print(f'{br}Testing NeuralNetwork for regression{br}')
    #---------------------------------------------------------------#
    X = np.arange(100).reshape((-1, 1))
    T = np.sin(X * 0.04)

    n_hiddens_list = [10, 10]

    # nnet = NeuralNetwork(X.shape[1], n_hiddens_list,
    #                      T.shape[1], activation_f='tanh')
    # nnet.summary()
    # nnet.train(X, T, n_epochs=1000, batch_size=32,
    #            learning_rate=0.01, opt='sgd')
    # Y = nnet.use(X)

    # print(f'RMSE: {rmse(T, Y):.3f}')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()

    # print(f'{br}Testing NeuralNetwork for CNN regression{br}')
    # #---------------------------------------------------------------#
    # # TODO: requires C, H, W dimensions
    # X = np.zeros((100, 1, 10, 10))
    # T = np.zeros((100, 1))
    # for i in range(100):
    #     col = i // 10
    #     X[i, :, 0:col + 1, 0] = 1
    #     T[i, 0] = col + 1

    # conv_layers = [{'n_units': 1, 'shape': [3, 3]},
    #                {'n_units': 1, 'shape': [3, 3]}]
    # n_hiddens_list = [10]

    # nnet = NeuralNetwork(X.shape[1:], n_hiddens_list,
    #                      T.shape[1], conv_layers, activation_f='tanh')
    # nnet.summary()
    # nnet.train(X, T, n_epochs=1000, batch_size=32,
    #            learning_rate=0.001, opt='adam')
    # Y = nnet.use(X)

    # print(f'RMSE: {rmse(T, Y):.3f}')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()

    # print(f'{br}Testing NeuralNetwork for CNN classification{br}')
    # #---------------------------------------------------------------#
    # X = np.zeros((100, 1, 10, 10))
    # T = np.zeros((100, 1))
    # for i in range(100):
    #     col = i // 10
    #     X[i, 0, :, 0:col + 1] = 1
    #     # TODO: class must be between [0, num_classes-1]
    #     T[i, 0] = 0 if col < 3 else 1 if col < 7 else 2

    # n_hiddens_list = [5]*2
    # conv_layers = [{'n_units': 3, 'shape': 3},
    #                {'n_units': 1, 'shape': [3, 3]}]

    # nnet = NeuralNetworkClassifier(X.shape[1:], n_hiddens_list, len(
    #     np.unique(T)), conv_layers, use_gpu=True, seed=None)
    # nnet.summary()
    # nnet.train(X, T, validation_data=None,
    #            n_epochs=50, batch_size=32, learning_rate=0.01, opt='adam',  # accsgd
    #            ridge_penalty=0, verbose=True)
    # Y = nnet.use(X)
    # print(f'Accuracy: {accuracy(Y, T)}:.3f')
    # # plt.plot(nnet.train_error_trace)
    # # plt.show()
