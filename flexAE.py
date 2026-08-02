import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import copy
import math
#import seaborn as sns
import warnings

from sklearn.neighbors import NearestNeighbors

import torch
from torch import nn, optim
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader,  TensorDataset, random_split
from torch import optim





class flexAE(nn.Module): #AE is child of nn.Module class. It is base class for all neural network modules
    '''

    '''

    def __init__(self, input_dim = 6, head_dim = [8], hidden_dim = [16,8], latent_dim = 4, classifier_dim=[8, 8], include_variational = False, include_scatter = False, include_error_head = False, include_classifier = True):

        ### Setup ###
        
        super().__init__() #This makes sure to initialize the parent class too


        self.include_variational = include_variational
        self.include_scatter = include_scatter
        self.include_error_head = include_error_head
        self.include_classifier = include_classifier

        

        if type(hidden_dim) in [int,float]: #Make hidden dim a list if not already
            hidden_dim = [hidden_dim]

        if type(head_dim) in [int,float]:#Make hidden dim a list if not already
            head_dim = [head_dim]

        if type(classifier_dim) in [int,float]:#Make hidden dim a list if not already
            classifier_dim = [classifier_dim]
            
        
        self.input_dim = input_dim # input dim depends on dataset
        self.head_dim = head_dim # head dim is size for input heads
        self.hidden_dim = hidden_dim # hidden dim for first linear layer
        self.latent_dim = latent_dim # dimension in latent space
        self.classifier_dim = classifier_dim # dimension in classifier
        
        self.mse_loss_per_epoch = []
        self.kld_loss_per_epoch = []
        self.bce_loss_per_epoch = []
        self.total_loss_per_epoch = []

        self.alpha_per_epoch = []
        self.beta_per_epoch = []
        self.gamma_per_epoch = []

        self.validation_mse_loss_per_epoch = []
        self.validation_kld_loss_per_epoch = []
        self.validation_bce_loss_per_epoch = []
        self.validation_loss_per_epoch = []
        
        self.lr_per_epoch = []
        
        self.feature_residuals_mid_per_epoch = []
        self.feature_residuals_lower_per_epoch = []
        self.feature_residuals_upper_per_epoch = []

        self.feature_scatter_mid_per_epoch = []
        self.feature_scatter_lower_per_epoch = []
        self.feature_scatter_upper_per_epoch = []

        self.mu_mid_per_epoch = []
        self.mu_lower_per_epoch = []
        self.mu_upper_per_epoch = []

        self.logvar_mid_per_epoch = []
        self.logvar_lower_per_epoch = []
        self.logvar_upper_per_epoch = []

        self.latent_map = None
        self.scaler = None

        self.save = False #Default false, only become true if activated
        self.save_dir = ''
        self.model_name = ''
        self.save_last = False
        self.save_best = False
        self.save_every_n_epochs = np.inf


        
        if self.include_error_head:
            # Define Value Head
            value_head_lst = []
            i_dim = input_dim
            for i, h_dim in enumerate(head_dim):
                hidden_layer = nn.Sequential(
                    nn.Linear(i_dim, h_dim),
                    #nn.ReLU()
                    nn.GELU()
                )
                
                value_head_lst.append(hidden_layer)
                i_dim = h_dim
            
            # value_head_output = nn.Sequential(nn.BatchNorm1d(head_dim[-1])) # Add last value head batchnorm layer
            # value_head_lst.append(value_head_output)
            
            self.value_head = nn.Sequential(*value_head_lst)
                
    
    
            # Define Error Head
            error_head_lst = []
            i_dim = input_dim
            for i, h_dim in enumerate(head_dim):
                hidden_layer = nn.Sequential(
                    nn.Linear(i_dim, h_dim),
                    #nn.ReLU()
                    nn.GELU()
                    #nn.LeakyReLU(negative_slope=0.5)
                )
                error_head_lst.append(hidden_layer)
                i_dim = h_dim
    
            # error_head_output = nn.Sequential(nn.BatchNorm1d(head_dim[-1])) # Add last error head batchnorm layer
            # error_head_lst.append(error_head_output)
            
            self.error_head = nn.Sequential(*error_head_lst)

        

        # Define Encoder
        encoder_lst = []
        if self.include_error_head:
            i_dim = head_dim[-1] * 2 #The input here is the concatenated output of the value and error heads. 
        else:
            i_dim = input_dim
        for i, h_dim in enumerate(hidden_dim):
            hidden_layer = nn.Sequential(
                nn.Linear(i_dim, h_dim),
                #nn.ReLU()
                nn.GELU()
            )
            
            encoder_lst.append(hidden_layer)
            i_dim = h_dim
            
        if self.include_variational == False:
            encoder_output = nn.Sequential(nn.Linear(hidden_dim[-1], latent_dim)) # Add last decoder layer
            encoder_lst.append(encoder_output)

        self.encoder = nn.Sequential(*encoder_lst)
        
    
        # Define VAE mu and logvar
        if self.include_variational: 
            self.latent_mu = nn.Linear(hidden_dim[-1], latent_dim) # Note that mu and logvar are technically not the latent space, but they define the latent space z 
            self.latent_logvar = nn.Linear(hidden_dim[-1], latent_dim)


        # Define Decoder 
        decoder_lst = []
        i_dim = latent_dim
        for i, h_dim in enumerate(hidden_dim[::-1]):
            hidden_layer = nn.Sequential(
                nn.Linear(i_dim, h_dim),
                #nn.ReLU()
                nn.GELU()
            )
            decoder_lst.append(hidden_layer)
            i_dim = h_dim
            
        decoder_output = nn.Sequential(nn.Linear(hidden_dim[0], input_dim)) # Add last decoder layer
        decoder_lst.append(decoder_output)
        
        self.decoder = nn.Sequential(*decoder_lst)


        # Define Scatter 
        if self.include_scatter:
            scatter_lst = []
            i_dim = latent_dim
            for i, h_dim in enumerate(hidden_dim[::-1]):
                hidden_layer = nn.Sequential(
                    nn.Linear(i_dim, h_dim),
                    #nn.ReLU()
                    nn.GELU()
                )
                scatter_lst.append(hidden_layer)
                i_dim = h_dim
                
            scatter_output = nn.Sequential(
                nn.Linear(hidden_dim[0], input_dim),# Add last decoder layer
                nn.Sigmoid() #And sigmoid to enforce positivity
            ) 
            scatter_lst.append(scatter_output)
    
            self.scatter = nn.Sequential(*scatter_lst)
    


        # Define Classifier
        if self.include_classifier:
            classifier_lst = []
            i_dim = latent_dim #potentially do *2 if I also want to include the logvar uncertainties in the classifier? Though in that case I would need to adjust the self.classify too
            for i, h_dim in enumerate(classifier_dim):
                hidden_layer = nn.Sequential(
                    nn.Linear(i_dim, h_dim),
                    #nn.ReLU()
                    nn.GELU(),
                    # nn.Dropout(0.1)
                )
                classifier_lst.append(hidden_layer)
                i_dim = h_dim
    
            classifier_output = nn.Sequential(
                nn.Linear(i_dim, 1),# Add last layer
                nn.Sigmoid() #And sigmoid to have output between 0 and 1
            ) 
            classifier_lst.append(classifier_output)
    
            self.classifier = nn.Sequential(*classifier_lst)
        


    ### =========== ###
    ### VAE methods ###
    ### =========== ###


    def encode(self, x, x_err = None):

        # Pass through heads
        if self.include_error_head:
            x_err = torch.log10(x_err)
            values = self.value_head(x)
            errors = self.error_head(x_err)
            x = torch.cat([values, errors], dim=1)

        
        # Pass through main encoder
        h = self.encoder(x)

        encoder_output = {}
        if self.include_variational:
        # Map to latent space parameters mu and logvar
            mu, logvar = self.latent_mu(h), self.latent_logvar(h)
            encoder_output['mu'] = mu
            encoder_output['logvar'] = logvar
        else:
            encoder_output['mu'] = h
        
        return encoder_output

    
    def decode(self, x):
        h = self.decoder(x)
        return h

    def intrinsic_scatter(self, x):
        h = self.scatter(x)
        return h
    

    def reparameterize(self, mu, logvar):
        """Sample from Gaussian posterior of z with reparametrization"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def classify(self, mu):
        """Use the MLP to classify based on latent space"""
        return self.classifier(mu)
        #combined = torch.cat([mu, logvar], dim=1) #activate this if I want classifier to also look at logvars.
        # return self.classifier(combined)
        
        

    def forward(self, x, x_err = None):
        """
        This is changed compared to VAE
        """
        encoder_output = self.encode(x, x_err) # get posterior parameters
        
        if self.include_variational:
            z = self.reparameterize(encoder_output['mu'], encoder_output['logvar']) # sample with reparametrization
        else:
            z = encoder_output['mu']
        
        reconstructed_x = self.decode(z) # get likelihood parameters

        if self.include_scatter:
            scatter = self.intrinsic_scatter(z) # get scatter

        if self.include_classifier: 
            classification_prediction = self.classify(encoder_output['mu']) #get classification

        results = {}
        results['reconstructed_x'] = reconstructed_x
        if self.include_scatter:
            results['scatter'] = scatter
            
        results['mu'] = encoder_output['mu']

        if self.include_variational:   
            results['logvar'] = encoder_output['logvar']

        if self.include_classifier:
            results['classification_prediction'] = classification_prediction
        
        return results


    ### ==================== ###
    ### Loss-related methods ###
    ### ==================== ###



    def loss_function(self, x, x_err, results, true_labels, alpha = 1.0, beta=1.0, gamma = 1.0):
        """Computes the AE loss."""

        if self.include_scatter:
            weighted_mse_loss = self.get_weighted_mse_loss(x, x_err, results['reconstructed_x'], results['scatter'], penalty = True)
        else:
            weighted_mse_loss = self.get_weighted_mse_loss(x, x_err, results['reconstructed_x'], torch.zeros_like(x), penalty = False)
            

        if self.include_variational:
            kld_loss = self.get_KLD_loss(results['mu'], results['logvar'])
        else:
            kld_loss = torch.tensor(0.0, device=x.device)

        if self.include_classifier:
            bce_loss = self.get_BCE_loss(results['classification_prediction'], true_labels)
        else:
            bce_loss = torch.tensor(0.0, device=x.device)

        total_loss = alpha * weighted_mse_loss + beta * kld_loss + gamma * bce_loss
    
        return total_loss, weighted_mse_loss, kld_loss, bce_loss


    
    def get_mse_loss(self, x, reconstructed_x):
        '''
        MSE between data and reconstructed data
        '''
        return torch.mean(torch.sum(0.5 * (reconstructed_x - x)**2, dim=1), dim=0)
    
    
    def get_weighted_mse_loss(self, x, x_err, reconstructed_x, reconstructed_scatter, penalty = False):
        '''
        Weighted MSE between data and reconstructed data.
        Note that we add an extra penalty term at the end. This is to avoid the sVAE making the reconstructed_scatter 
        so large that the weighted MSE goes to 0. 
        '''
        weight = x_err**2 + reconstructed_scatter**2

        if penalty:
            return torch.mean(torch.sum(0.5*(reconstructed_x - x)**2./weight + 0.5*torch.log(weight), dim=1), dim=0)
        else:
            return torch.mean(torch.sum(0.5*(reconstructed_x - x)**2./weight, dim=1), dim=0)
            
        
    
    def get_KLD_loss(self, mu, logvar):
        '''
        Implements Kullback–Leibler divergence.
    
        KL divergence between the posterior and the prior (both Gaussian)
        Measures how much the latent distribution deviates from a unit Gaussian
        Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        '''
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)


    def get_BCE_loss(self, predictions, labels):
        '''
        Advantage of BCE loss is that it heavily penalizes confident but incorrect predictions. 
        '''
        labels = labels.float().view(-1, 1) #To make sure they go from [X] to [X,1]
        criterion = nn.BCELoss()
        bce_loss = criterion(predictions, labels)
        return bce_loss



    ### ================ ###
    ### Plotting methods ###
    ### ================ ###

    def plot_loss_per_epoch(self, figsize = (8,4), losses = ['total','validation','mse','bce','val_bce','kld'], yscale = 'linear', ax = None):

        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)

        if 'total' in losses and len(self.total_loss_per_epoch) > 0:
            plt.plot(self.total_loss_per_epoch, label = 'Total Loss')
            
        if 'validation' in losses and len(self.validation_loss_per_epoch) > 0:
            plt.plot(self.validation_loss_per_epoch, label = 'Validation Loss')
            
        if 'mse' in losses and len(self.mse_loss_per_epoch) > 0:
            plt.plot(self.mse_loss_per_epoch, label = 'MSE Loss')

        if 'val_mse' in losses and len(self.validation_mse_loss_per_epoch) > 0:
            plt.plot(self.validation_mse_loss_per_epoch, label = 'Validation MSE Loss')

        if 'bce' in losses and len(self.bce_loss_per_epoch) > 0:
            plt.plot(self.bce_loss_per_epoch, label = 'BCE Loss')

        if 'val_bce' in losses and len(self.validation_bce_loss_per_epoch) > 0:
            plt.plot(self.validation_bce_loss_per_epoch, label = 'Validation BCE Loss')
            
        if 'kld' in losses and len(self.kld_loss_per_epoch) > 0:
            plt.plot(self.kld_loss_per_epoch, label = 'KLD Loss')

        if 'val_kld' in losses and len(self.validation_kld_loss_per_epoch) > 0:
            plt.plot(self.validation_kld_loss_per_epoch, label = 'Validation KLD Loss')

        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale(yscale)

        if ax == None:
            plt.show()





    def plot_loss_stackplot(self, figsize = (8,4), colors = ['#3498db', '#e74c3c', '#2ecc71'], ax = None):

        epochs = list(range(len(self.mse_loss_per_epoch)))
        
        mse = np.abs(np.array(self.mse_loss_per_epoch))
        bce = np.abs(np.array(self.bce_loss_per_epoch))
        kld = np.abs(np.array(self.kld_loss_per_epoch))

        alpha = np.array(self.alpha_per_epoch)
        beta = np.array(self.beta_per_epoch)
        gamma = np.array(self.gamma_per_epoch)

        bce = bce * gamma
        kld = kld * beta
        mse = mse * alpha
        
        # 1. Calculate the total list (the fourth list)
        total_list = mse + bce + kld
        
        # 2. Calculate the fraction of each list
        # Use np.divide to handle potential division by zero if necessary
        mse_frac = mse / total_list
        bce_frac = bce / total_list
        kld_frac = kld / total_list
        
        # 3. Create the stacked area plot
        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)
        
        
        plt.stackplot(epochs, mse_frac, bce_frac, kld_frac,
                      labels=['MSE', 'BCE', 'KLD'],
                      colors=colors)
        
        plt.xlabel('Epoch')
        plt.ylabel('Fraction of Total Loss')
        plt.legend(loc='upper right')
        plt.ylim(0, 1)
        plt.xlim(epochs[0], epochs[-1])

        if ax == None:
            plt.show()
    





    def plot_learningrate_per_epoch(self, figsize = (8,4), ax = None):

        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)
       

        plt.plot(self.lr_per_epoch)

        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')

        if ax == None:
            plt.show()



    def plot_feature_residuals_per_epoch(self, figsize = (8,4), labels = [], xlim = [], ylim = [], yscale = 'linear', ax = None):

        feature_residuals_mid = np.array(self.feature_residuals_mid_per_epoch).T
        # feature_residuals_lower = np.array(self.feature_residuals_lower_per_epoch).T
        # feature_residuals_upper = np.array(self.feature_residuals_upper_per_epoch).T
        epochs = list(range(len(feature_residuals_mid[0])))
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_residuals_mid))]

        
        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)


        for i,fr in enumerate(feature_residuals_mid):
            plt.plot(epochs, fr, label = labels[i])
            #plt.fill_between(epochs, feature_residuals_lower[i], feature_residuals_upper[i], alpha = 0.1)

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Feature Residuals')
        plt.yscale(yscale)

        if len(xlim) > 0:
            plt.xlim(xlim)

        if len(ylim) > 0:
            plt.ylim(ylim)



        if ax == None:
            plt.show()




    def plot_feature_residuals_per_epoch_v2(self, figsize = (8,8), labels = [], xlim = [], ylim = [], yscale = 'linear', ncols = 2):

        feature_residuals_mid = np.array(self.feature_residuals_mid_per_epoch).T
        feature_residuals_lower = np.array(self.feature_residuals_lower_per_epoch).T
        feature_residuals_upper = np.array(self.feature_residuals_upper_per_epoch).T
        epochs = list(range(len(feature_residuals_mid[0])))
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_residuals_mid))]

        
        nfigs = len(feature_residuals_mid)
        nrows = math.ceil(nfigs/ncols)
        
        plt.figure(figsize = figsize)


        for i,fr in enumerate(feature_residuals_mid):
            
            plt.subplot(nrows,ncols, i+1)

            plt.plot(epochs, fr, label = labels[i])
            plt.fill_between(epochs, feature_residuals_lower[i], feature_residuals_upper[i], alpha = 0.4)

            plt.title(labels[i])
            
            plt.xlabel('Epoch')
            #plt.ylabel('Feature Residuals')
            plt.yscale(yscale)

            if len(xlim) > 0:
                plt.xlim(xlim)

            if len(ylim) > 0:
                plt.ylim(ylim)

        plt.tight_layout()
        plt.show()




    def plot_feature_residuals(self, figsize = (4,3), labels = [], yscale = 'linear', ax = None):
        
        feature_residuals_mid = np.array(self.feature_residuals_mid_per_epoch)
        feature_residuals_lower = np.array(self.feature_residuals_lower_per_epoch)
        feature_residuals_upper = np.array(self.feature_residuals_upper_per_epoch)

        lerr = feature_residuals_mid[-1] - feature_residuals_lower[-1]
        uerr = feature_residuals_upper[-1] - feature_residuals_mid[-1]
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_residuals_mid[0]))]

        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)

        
        plt.bar(labels, feature_residuals_mid[-1], yerr = [lerr, uerr])

        plt.ylabel('Residuals')
        plt.yscale(yscale)

        if ax == None:
            plt.show()




    def plot_feature_scatter_per_epoch(self, figsize = (8,4), labels = [], xlim = [], ylim = [], yscale = 'linear', ax = None):

        feature_scatter_mid = np.array(self.feature_scatter_mid_per_epoch).T
        # feature_scatter_lower = np.array(self.feature_scatter_lower_per_epoch).T
        # feature_scatter_upper = np.array(self.feature_scatter_upper_per_epoch).T
        epochs = list(range(len(feature_scatter_mid[0])))
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_scatter_mid))]

        
        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)

        for i,fr in enumerate(feature_scatter_mid):
            plt.plot(epochs, fr, label = labels[i])
            #plt.fill_between(epochs, feature_scatter_lower[i], feature_scatter_upper[i], alpha = 0.1)

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Feature Scatter')
        plt.yscale(yscale)

        if len(xlim) > 0:
            plt.xlim(xlim)

        if len(ylim) > 0:
            plt.ylim(ylim)
            

        if ax == None:
            plt.show()




    def plot_feature_scatter_per_epoch_v2(self, figsize = (8,8), labels = [], xlim = [], ylim = [], yscale = 'linear', ncols = 2):

        feature_scatter_mid = np.array(self.feature_scatter_mid_per_epoch).T
        feature_scatter_lower = np.array(self.feature_scatter_lower_per_epoch).T
        feature_scatter_upper = np.array(self.feature_scatter_upper_per_epoch).T
        epochs = list(range(len(feature_scatter_mid[0])))
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_scatter_mid))]

        
        nfigs = len(feature_scatter_mid)
        nrows = math.ceil(nfigs/ncols)
        
        plt.figure(figsize = figsize)


        for i,fr in enumerate(feature_scatter_mid):
            
            plt.subplot(nrows,ncols, i+1)

            plt.plot(epochs, fr, label = labels[i])
            plt.fill_between(epochs, feature_scatter_lower[i], feature_scatter_upper[i], alpha = 0.4)

            plt.title(labels[i])
            
            plt.xlabel('Epoch')
            #plt.ylabel('Feature Scatter')
            plt.yscale(yscale)

            if len(xlim) > 0:
                plt.xlim(xlim)

            if len(ylim) > 0:
                plt.ylim(ylim)

        plt.tight_layout()
        plt.show()



    def plot_feature_scatter(self, figsize = (4,3), labels = [], yscale = 'linear', ax = None):
        
        feature_scatter_mid = np.array(self.feature_scatter_mid_per_epoch)
        feature_scatter_lower = np.array(self.feature_scatter_lower_per_epoch)
        feature_scatter_upper = np.array(self.feature_scatter_upper_per_epoch)

        lerr = feature_scatter_mid[-1] - feature_scatter_lower[-1]
        uerr = feature_scatter_upper[-1] - feature_scatter_mid[-1]
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(len(feature_scatter_mid[0]))]

        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)
        
        
        plt.bar(labels, feature_scatter_mid[-1], yerr = [lerr, uerr])

        plt.ylabel('Scatter')
        plt.yscale(yscale)

        if ax == None:
            plt.show()






    def compare_reconstructed_features(self, dataset, n_examples = 10, ncols = 5, labels = [], device = None, tick_ha = 'center'):
        
        if len(labels) == 0:
            labels = [f'Feature {i}' for i in range(self.input_dim)]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # if self.include_error_head == False and isinstance(dataset, TensorDataset) and len(dataset.tensors) == 2: # If user does not want errors and none are provided in the train_dataset, just add a dummy tensor. It'll get ignored later but this way code doesn't break
        #     data, data_labels = dataset.tensors
        #     dummy_err = torch.ones_like(data)
        #     dataset = TensorDataset(data, dummy_err, data_labels)

        dataset = self.check_tensordataset_shape(dataset)
    
        data_lst = []
        data_error_lst = []
        reconstructed_lst = []
        reconstructed_err_lst = []

        idx_examples = np.random.choice(range(len(dataset)), size = n_examples)

        self.eval()
        with torch.no_grad():

            for idx_example in idx_examples:
                data, data_err, label = dataset[idx_example] #random choice I made
                results = self(data.unsqueeze(0).to(device), data_err.unsqueeze(0).to(device))
            
                data = data.squeeze(0).detach().cpu().numpy()
                data_err = data_err.squeeze(0).detach().cpu().numpy()
                reconstructed_data = results['reconstructed_x'].squeeze(0).detach().cpu().numpy()

                if self.include_scatter:
                    reconstructed_err = results['scatter'].squeeze(0).detach().cpu().numpy()
                else:
                    reconstructed_err = np.ones_like(reconstructed_data)

                data_lst.append(data)
                data_error_lst.append(data_err)
                reconstructed_lst.append(reconstructed_data)
                reconstructed_err_lst.append(reconstructed_err)




        nrows = math.ceil(n_examples/ncols)

        plt.figure(figsize = (ncols*5, nrows*4))

        for i in range(n_examples):
            plt.subplot(nrows,ncols,i+1)
            plt.plot(labels, data_lst[i], label = 'Original', color = 'C0')
            if self.include_error_head:
                plt.fill_between(labels, data_lst[i] - data_error_lst[i], data_lst[i] + data_error_lst[i], alpha = 0.2, color = 'C0')
            
            plt.plot(labels, reconstructed_lst[i], ls = '--', label = 'Reconstructed', color = 'C1')

            if self.include_scatter:
                plt.fill_between(labels, reconstructed_lst[i] - reconstructed_err_lst[i], reconstructed_lst[i] + reconstructed_err_lst[i], alpha = 0.2, color = 'C1')
            plt.legend()
            plt.xticks(rotation=45, ha = tick_ha)

        plt.tight_layout()
        plt.show()





    def plot_mu_per_epoch(self, figsize = (8,4), labels = [], xlim = [], ylim = [], yscale = 'linear', ax = None):

        mu_mid = np.array(self.mu_mid_per_epoch).T
        mu_lower = np.array(self.mu_lower_per_epoch).T
        mu_upper = np.array(self.mu_upper_per_epoch).T
        epochs = list(range(len(mu_mid[0])))
        
        if len(labels) == 0:
            labels = [f'z{i}' for i in range(len(mu_mid))]

        
        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)


        for i, mu in enumerate(mu_mid):
            plt.plot(epochs, mu, label = labels[i])
            #plt.fill_between(epochs, mu_lower[i], mu_upper[i], alpha = 0.1)

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('mu')
        plt.yscale(yscale)

        if len(xlim) > 0:
            plt.xlim(xlim)

        if len(ylim) > 0:
            plt.ylim(ylim)
            
        if ax == None:
            plt.show()


    def plot_logvar_per_epoch(self, figsize = (8,4), labels = [], xlim = [], ylim = [], yscale = 'linear', ax = None):

        logvar_mid = np.array(self.logvar_mid_per_epoch).T
        logvar_lower = np.array(self.logvar_lower_per_epoch).T
        logvar_upper = np.array(self.logvar_upper_per_epoch).T
        epochs = list(range(len(logvar_mid[0])))
        
        if len(labels) == 0:
            labels = [f'logvar{i}' for i in range(len(logvar_mid))]

        
        if ax == None:
            plt.figure(figsize = figsize)
        else:
            plt.sca(ax)
        

        for i, logvar in enumerate(logvar_mid):
            plt.plot(epochs, logvar, label = labels[i])
            #plt.fill_between(epochs, logvar_lower[i], logvar_upper[i], alpha = 0.1)

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Logvar')
        plt.yscale(yscale)

        if len(xlim) > 0:
            plt.xlim(xlim)

        if len(ylim) > 0:
            plt.ylim(ylim)
            
        if ax == None:
            plt.show()



    def parameter_summary(self, return_total = False, print_summary = True):
        total_params = 0
        if print_summary:
            print(f"{'Layer Name':<30} | {'Parameters':<15}")
            print("-" * 50)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            if print_summary:
                print(f"{name:<30} | {params:<15,}")
            total_params += params
        if print_summary:
            print("-" * 50)
            print(f"{'Total Trainable Params':<30} | {total_params:<15,}")
        if return_total:
            return total_params
    


    
    
    ### ================ ###
    ### Training methods ###
    ### ================ ###



    def check_tensordataset_shape(self, tensor_dataset):

        if self.include_error_head == False and self.include_classifier == False:

            assert len(tensor_dataset.tensors) == 1, "Incorrect number of tensors in dataset. If `include_error_head = False` and `inlude_classifier = False`, then there should only be one tensor (the data) in the dataset."

            data = tensor_dataset.tensors[0]
            dummy_err = torch.ones_like(data)
            dummy_labels = torch.ones_like(data)/2
            tensor_dataset = TensorDataset(data, dummy_err, dummy_labels)


        if self.include_error_head == True and self.include_classifier == False:
        
            assert len(tensor_dataset.tensors) == 2, "Incorrect number of tensors in dataset. If `include_error_head = True` and `inlude_classifier = False`, then there should be two tensors (the data and the uncertainty on the data) in the dataset."

            data, data_err = tensor_dataset.tensors
            dummy_labels = torch.ones_like(data)/2
            tensor_dataset = TensorDataset(data, data_err, dummy_labels)


        if self.include_error_head == False and self.include_classifier == True:
        
            assert len(tensor_dataset.tensors) == 2, "Incorrect number of tensors in dataset. If `include_error_head = False` and `inlude_classifier = True`, then there should be two tensors (the data and the labels) in the dataset."

            data, labels = tensor_dataset.tensors
            dummy_err = torch.ones_like(data)
            tensor_dataset = TensorDataset(data, dummy_err, labels)



        if self.include_error_head == True and self.include_classifier == True:
        
            assert len(tensor_dataset.tensors) == 3, "Incorrect number of tensors in dataset. If `include_error_head = False` and `inlude_classifier = True`, then there should be three tensors (the data, the uncertainty on the data, and the labels) in the dataset."

            data, data_err, labels = tensor_dataset.tensors
            tensor_dataset = TensorDataset(data, data_err, labels)


        return tensor_dataset


    def KL_annealing(self, epoch, start_epoch, end_epoch, target_beta):
        if epoch < start_epoch:
            return 0.0
        if epoch >= end_epoch:
            return target_beta
        
        # Linear interpolation between start and end epochs
        return target_beta * (epoch - start_epoch) / (end_epoch - start_epoch)



    def get_residuals(self, train_loader, device):
        self.eval() #Set model to evaluate mode
        batch_residuals = []
        with torch.no_grad():
            for batch_idx, (data, data_err, true_labels) in enumerate(train_loader):
                # Forward Pass
                results = self(data.to(device), data_err.to(device))

                # Calculate residuals
                #batch_res = (torch.abs(data - reconstructed_batch)/data_err).cpu().numpy()
                batch_res = torch.abs(data - results['reconstructed_x']).cpu().numpy()
                batch_residuals.append(batch_res)
        
        epoch_residuals = np.vstack(batch_residuals)
        self.feature_residuals_mid_per_epoch.append(np.median(epoch_residuals, axis=0))
        self.feature_residuals_lower_per_epoch.append(np.percentile(epoch_residuals, 25, axis=0))
        self.feature_residuals_upper_per_epoch.append(np.percentile(epoch_residuals, 75, axis=0))



    def get_scatter(self, train_loader, device):
        self.eval() #Set model to evaluate mode
        batch_scatter = []
        with torch.no_grad():
            for batch_idx, (data, data_err, true_labels) in enumerate(train_loader):
                # Forward Pass
                results = self(data.to(device), data_err.to(device))

                # Calculate scatter
                batch_s = results['scatter'].cpu().numpy()
                batch_scatter.append(batch_s)
        
        epoch_scatter = np.vstack(batch_scatter)
        self.feature_scatter_mid_per_epoch.append(np.median(epoch_scatter, axis=0))
        self.feature_scatter_lower_per_epoch.append(np.percentile(epoch_scatter, 25, axis=0))
        self.feature_scatter_upper_per_epoch.append(np.percentile(epoch_scatter, 75, axis=0))




    def get_latent_space_params(self,train_loader, device):
        self.eval() #Set model to evaluate mode
        batch_mu = []
        batch_logvar = []
        with torch.no_grad():
            for batch_idx, (data, data_err, true_labels) in enumerate(train_loader):
                # Forward Pass
                results = self(data.to(device), data_err.to(device))
                
                # Calculate mu
                mu = results['mu'].cpu().numpy()
                batch_mu.append(mu)

                if self.include_variational:
                    # Calculate logvar
                    logvar = results['logvar'].cpu().numpy()
                    batch_logvar.append(logvar)

        

        epoch_mu = np.vstack(batch_mu)
        self.mu_mid_per_epoch.append(np.median(epoch_mu, axis=0))
        self.mu_lower_per_epoch.append(np.percentile(epoch_mu, 25, axis=0))
        self.mu_upper_per_epoch.append(np.percentile(epoch_mu, 75, axis=0))

        if self.include_variational:
            epoch_logvar = np.vstack(batch_logvar)
            self.logvar_mid_per_epoch.append(np.median(epoch_logvar, axis=0))
            self.logvar_lower_per_epoch.append(np.percentile(epoch_logvar, 25, axis=0))
            self.logvar_upper_per_epoch.append(np.percentile(epoch_logvar, 75, axis=0))

        


    def validate_epoch(self, val_loader, device, alpha = 1.0, beta = 1.0, gamma = 1.0):
        """Validate the VAE for one epoch"""
        self.eval() #Set model to evaluate mode
        val_loss = 0
        val_mse_loss = 0
        val_kld_loss = 0
        val_bce_loss = 0

        with torch.no_grad():
            for batch_idx, (data, data_err, true_labels) in enumerate(val_loader):
                # Forward Pass
                results = self(data.to(device), data_err.to(device))
                loss, mse, kld, bce = self.loss_function(data.to(device), data_err.to(device), results, true_labels, alpha = alpha, beta = beta, gamma = gamma)
                
                val_loss += loss.item()
                val_mse_loss += mse.item()
                val_bce_loss += bce.item()
                val_kld_loss += kld.item()
                
        val_loss = val_loss / (batch_idx+1)
        val_mse_loss = val_mse_loss / (batch_idx+1)
        val_bce_loss = val_bce_loss / (batch_idx+1)
        val_kld_loss = val_kld_loss / (batch_idx+1)

        self.validation_loss_per_epoch.append(val_loss)
        self.validation_mse_loss_per_epoch.append(val_mse_loss)
        self.validation_bce_loss_per_epoch.append(val_bce_loss)
        self.validation_kld_loss_per_epoch.append(val_kld_loss)



    def train_epoch(self, train_loader, optimizer, device, alpha = 1.0, beta = 1.0, gamma = 1.0):
        """Trains the AE for one epoch."""
        self.train() # Set model to training mode
        train_loss = 0
        mse_loss = 0
        kld_loss = 0
        bce_loss = 0
    
        for batch_idx, (data, data_err, true_labels) in enumerate(train_loader):
            optimizer.zero_grad() # Clear previous gradients
    
            # Forward Pass
            results = self(data.to(device), data_err.to(device))
    
            # Calculate Loss
            loss, mse, kld, bce = self.loss_function(data.to(device), data_err.to(device), results, true_labels, alpha = alpha, beta = beta, gamma = gamma)
    
            # Backward Pass
            loss.backward() #The backpropagation step. Calculates the gradient. 
            kld_loss += kld.item()
            mse_loss += mse.item()
            bce_loss += bce.item()
            train_loss += loss.item() # Add the loss
            
            optimizer.step() # Update weights 
    
        train_loss = train_loss / (batch_idx+1) #Calculate average loss per batch
        mse_loss = mse_loss / (batch_idx+1) #Calculate average loss per batch
        kld_loss = kld_loss / (batch_idx+1) #Calculate average loss per batch
        bce_loss = bce_loss / (batch_idx+1) #Calculate average loss per batch
    
    
        self.total_loss_per_epoch.append(train_loss)
        self.mse_loss_per_epoch.append(mse_loss)
        self.kld_loss_per_epoch.append(kld_loss)
        self.bce_loss_per_epoch.append(bce_loss)

        self.alpha_per_epoch.append(alpha)
        self.beta_per_epoch.append(beta)
        self.gamma_per_epoch.append(gamma)
    
        return train_loss

    
    

    def train_model(self, train_dataset, epochs = 100, learning_rate = 1e-3, batch_size = 32, beta = 0.1, gamma = 1.0, validation_split = 0.0, kl_annealing_epochs = [200,500], optimizer = None, scheduler = None, device = None):
        ### Setup ###
        # if self.include_error_head == False and isinstance(train_dataset, TensorDataset) and len(train_dataset.tensors) == 2: # If user does not want errors and none are provided in the train_dataset, just add a dummy tensor. It'll get ignored later but this way code doesn't break
        #     data, labels = train_dataset.tensors
        #     dummy_err = torch.ones_like(data)
        #     train_dataset = TensorDataset(data, dummy_err, labels)

        train_dataset = self.check_tensordataset_shape(train_dataset)


        if validation_split > 0.0:
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [1.0-validation_split,validation_split],
                generator=torch.Generator().manual_seed(30) # For reproducability
            )
            
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if scheduler is None: 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epochs//10, factor=0.5)


        if self.include_variational == False:
            beta = 0
        if self.include_classifier == False:
            gamma = 0
        
        #### Train the model ###
        for epoch in tqdm(range(1, epochs + 1)):

            if self.include_variational:
                beta_current = self.KL_annealing(epoch, start_epoch=kl_annealing_epochs[0], end_epoch=kl_annealing_epochs[1], target_beta = beta)#0.0001
            else:
                beta_current = 0

            
            epoch_loss = self.train_epoch(train_loader, optimizer, device, beta = beta_current, gamma = gamma)


            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()
            else:
                scheduler.step()
                
            #scheduler.step(epoch_loss)
            #scheduler.step(epoch)
            
            self.lr_per_epoch.append(optimizer.param_groups[0]['lr'])
            
            if validation_split > 0.0:
                self.validate_epoch(val_loader, device, beta = beta_current, gamma = gamma)


            # TODO: combine these three?
            self.get_residuals(train_loader, device)
            if self.include_scatter:
                self.get_scatter(train_loader, device)
            self.get_latent_space_params(train_loader, device)

            self.consider_saving_model()



        self.build_latent_map(train_loader, n_neighbors = 10, device = device)
        self.consider_saving_model(last_epoch = True)




    def train_classifier(self, train_dataset, epochs=1000, learning_rate=1e-3, batch_size=32, gamma=1.0, validation_split=0.1, device=None, optimizer = None, scheduler = None):
        if not self.include_classifier:
            print("Error: Model was not initialized with a classifier.")
            return

        # if self.include_error_head == False and isinstance(train_dataset, TensorDataset) and len(train_dataset.tensors) == 2: # If user does not want errors and none are provided in the train_dataset, just add a dummy tensor. It'll get ignored later but this way code doesn't break
        #     data, labels = train_dataset.tensors
        #     dummy_err = torch.ones_like(data)
        #     train_dataset = TensorDataset(data, dummy_err, labels)

        train_dataset = self.check_tensordataset_shape(train_dataset)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        for param in self.parameters(): #First, freeze all
            param.requires_grad = False

        for param in self.classifier.parameters(): #Unfreeze only the classifier
            param.requires_grad = True


        
        if optimizer is None:
            optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate) #Optimizer for classifier params only! 
        
        if scheduler is None: 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epochs//10, factor=0.5)
            
        

        ### Setup ###
        if validation_split > 0.0:
            train_dataset, val_dataset = random_split(
                train_dataset, 
                [1.0-validation_split,validation_split],
                generator=torch.Generator().manual_seed(30) # For reproducability
            )
            
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        


        # 4. Training Loop
        for epoch in tqdm(range(1, epochs + 1)):
            epoch_loss = self.train_epoch(train_loader, optimizer, device, alpha = 0, beta=0, gamma=gamma) ## We keep alpha and beta=0 because we aren't updating the latent space/VAE part anyway



            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()
            else:
                scheduler.step()
            #scheduler.step(epoch_loss)
            #scheduler.step(epoch)
            
            self.lr_per_epoch.append(optimizer.param_groups[0]['lr'])

            
            if validation_split > 0.0:
                self.validate_epoch(val_loader, device, alpha = 0, beta=0, gamma=gamma)


            self.consider_saving_model()

        # 5. Re-enable gradients for the whole model (optional, but good practice)
        for param in self.parameters():
            param.requires_grad = True


        
        self.build_latent_map(train_loader, n_neighbors = 10, device = device)
        self.consider_saving_model(last_epoch = True)


    
    def build_latent_map(self, train_loader, n_neighbors = 10, device = None, return_map = False):
        '''
        Build a Nearest Neighbor map of the latent space using the TensorDataset provided. Recommended to use TensorDatasets, not DataLoaders.
    
        Should work now with both pytorch DataLoaders and pytorch TensorDataset
        
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        '''
        
    
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.eval()
        all_mu = []
    
        
        is_dataloader = isinstance(train_loader, DataLoader) #Figure out if using dataloader or dataset

        if not is_dataloader:
            train_loader = self.check_tensordataset_shape(train_loader)
            
        
        with torch.no_grad():
            for data, data_err, _ in train_loader:
    
                if not is_dataloader: #If it is a TensorDataset, unsqueeze
                    data = data.unsqueeze(0)
                    data_err = data_err.unsqueeze(0)
                    
                encoder_output = self.encode(data.to(device), data_err.to(device))

                all_mu.append(encoder_output['mu'].cpu().numpy())

                
        
        all_mu = np.vstack(all_mu)
        
        latent_map = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        latent_map.fit(all_mu)
        
        self.latent_map = latent_map
        if return_map:
            return latent_map



    def get_latent_distance_score(self, data, data_err = None, n_neighbors = 10, device = None):
        '''
        Using the latent_map in the model, calculate the distance score of your new object.
        The latent map is first calculated using ``self.build_latent_map``. This is done automatically after traning.
        '''

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if data_err == None: #This is just a dummy variable. If include_error_head == False, then encoder will not look at this.
            data_err = torch.ones_like(data)


        self.eval()
        with torch.no_grad():
            encoder_output = self.encode(data.to(device), data_err.to(device))
            mu = encoder_output['mu'].cpu().numpy()

            
        # Find distances to the 10 nearest neighbors
        distances, _ = self.latent_map.kneighbors(mu, n_neighbors = n_neighbors)
        
        # Return the average distance across neighbors
        return np.mean(distances, axis=1)




    ### =========== ###
    ### I/O methods ###
    ### =========== ###


    def consider_saving_model(self, last_epoch = False):

        if self.save: 
            current_epoch = len(self.total_loss_per_epoch)
            
            if len(self.validation_loss_per_epoch) == 1:
                best_val_loss = np.inf
            else:
                best_val_loss = np.min(self.validation_loss_per_epoch[:-1]) #All losses except the newest one
            current_val_loss = self.validation_loss_per_epoch[-1]

            # If a new best model appears, save it.
            if self.save_best: 
                if current_val_loss < best_val_loss:
                    self.save_model(self.save_dir, self.model_name, file_name = 'best')
                    if self.save_last:
                        self.save_model(self.save_dir, self.model_name, file_name = 'last')

            # Save this epoch if it is multiple
            if current_epoch % self.save_every_n_epochs == 0:
                self.save_model(self.save_dir, self.model_name, file_name = f'epoch_{int(current_epoch)}')
                if self.save_last:
                    self.save_model(self.save_dir, self.model_name, file_name = 'last')

            # If it is the last epoch in the training cycle, save it too
            if last_epoch:
                self.save_model(self.save_dir, self.model_name, file_name = f'epoch_{int(current_epoch)}')
                if self.save_last:
                    self.save_model(self.save_dir, self.model_name, file_name = 'last')



                


    def setup_autosaving(self, save_dir = 'saved_models', model_name = 'lastest_model', save_last = True, save_best = False, save_every_n_epochs = np.inf):
        """
        Setup the autosaving process. 

        It will go to save_dir, and then create a new subdirectory for this model. Within
        that subdirectory, the models are saved.

        save_last : bool, default = True
            Whenever a new model state is saved, also save it and overwrite last.pt. This way, the user can always load last.pt and it will load the last model.

        save_best : bool, default = True
            Whenever a new best (= lowest validation loss) model is achieveed save it and overwrite best.pt. This way, the user can always load best.pt and it will load the best model.
        
        save_every_n_epochs : bool, default = np.inf
            Save the model after every n epochs. By default, will never save.
        """

        if save_dir[-1] != '/':
            save_dir += '/'

        if model_name[-1] != '/':
            model_name += '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(save_dir + model_name):
            os.makedirs(save_dir + model_name)

        self.save = True #Default false, only become true if activated here.
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_last = save_last
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs




    def save_model(self, save_dir = 'saved_models', model_name = 'lastest_model', file_name = ''):
        """
        Saves the complete state of the flexAE model to a single file.
        Includes architecture parameters, tracking histories, and weights.

        It will go to save_dir, and then create a new subdirectory for this model. Within
        that subdirectory, it will create a .pt file with all the information with file_name.
        """

        # 1. Gather all configuration parameters
        init_args = {
            'input_dim': self.input_dim,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'classifier_dim': self.classifier_dim,
            'include_variational': self.include_variational,
            'include_scatter': self.include_scatter,
            'include_error_head': self.include_error_head,
            'include_classifier': self.include_classifier
        }

        # 2. Gather all training history tracking metrics
        history_state = {
            'mse_loss_per_epoch': self.mse_loss_per_epoch,
            'kld_loss_per_epoch': self.kld_loss_per_epoch,
            'bce_loss_per_epoch': self.bce_loss_per_epoch,
            'total_loss_per_epoch': self.total_loss_per_epoch,
            'alpha_per_epoch': self.alpha_per_epoch,
            'beta_per_epoch': self.beta_per_epoch,
            'gamma_per_epoch': self.gamma_per_epoch,
            'validation_mse_loss_per_epoch': self.validation_mse_loss_per_epoch,
            'validation_kld_loss_per_epoch': self.validation_kld_loss_per_epoch,
            'validation_bce_loss_per_epoch': self.validation_bce_loss_per_epoch,
            'validation_loss_per_epoch': self.validation_loss_per_epoch,
            'lr_per_epoch': self.lr_per_epoch,
            'feature_residuals_mid_per_epoch': self.feature_residuals_mid_per_epoch,
            'feature_residuals_lower_per_epoch': self.feature_residuals_lower_per_epoch,
            'feature_residuals_upper_per_epoch': self.feature_residuals_upper_per_epoch,
            'feature_scatter_mid_per_epoch': self.feature_scatter_mid_per_epoch,
            'feature_scatter_lower_per_epoch': self.feature_scatter_lower_per_epoch,
            'feature_scatter_upper_per_epoch': self.feature_scatter_upper_per_epoch,
            'mu_mid_per_epoch': self.mu_mid_per_epoch,
            'mu_lower_per_epoch': self.mu_lower_per_epoch,
            'mu_upper_per_epoch': self.mu_upper_per_epoch,
            'logvar_mid_per_epoch': self.logvar_mid_per_epoch,
            'logvar_lower_per_epoch': self.logvar_lower_per_epoch,
            'logvar_upper_per_epoch': self.logvar_upper_per_epoch,
            'latent_map': self.latent_map,
            'scaler': self.scaler,
            'save' : self.save,
            'save_dir' : self.save_dir,
            'model_name' : self.model_name,
            'save_last' : self.save_last,
            'save_best' : self.save_best,
            'save_every_n_epochs' : self.save_every_n_epochs
        }

        # 3. Combine everything into a single checkpoint dictionary
        checkpoint = {
            'init_args': init_args,
            'history_state': history_state,
            'state_dict': self.state_dict()
        }


        # 4. File structure things and actually saving. 
        if save_dir[-1] != '/':
            save_dir += '/'

        if model_name[-1] != '/':
            model_name += '/'

        if file_name == '':
            file_name = 'last'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(save_dir + model_name):
            os.makedirs(save_dir + model_name)

        save_loc = save_dir + model_name + file_name + '.pt'
        torch.save(checkpoint, save_loc)



    @classmethod
    def load_model(cls, save_dir = 'saved_models', model_name = 'lastest_model', file_name = '', map_location=None):
        """
        Class method to instantiate and fully restore a flexAE model from a saved file.
        Usage: model = flexAE.load_model('my_model.pt')
        """
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if save_dir[-1] != '/':
            save_dir += '/'

        if model_name[-1] != '/':
            model_name += '/'


        if file_name == '':
            file_name = 'last'


        save_loc = save_dir + model_name + file_name + '.pt'

        checkpoint = torch.load(save_loc, map_location=map_location, weights_only = False)

        # 1. Instantiate a fresh model instance using the saved configuration arguments
        model = cls(**checkpoint['init_args'])

        # 2. Restore all the tracking histories and sklearn objects
        for key, value in checkpoint['history_state'].items():
            setattr(model, key, value)

        # 3. Load the structural neural network weights
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Model successfully loaded from {save_loc}.")
        return model
    






def split_tensordataset(dataset, train_fraction=0.8, seed=42):

    generator = torch.Generator().manual_seed(seed)
    
    num_samples = len(dataset)
    train_size = int(train_fraction * num_samples)
    
    # Generate random permutation of indices
    perm = torch.randperm(num_samples, generator=generator)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    
    # Slice the underlying tensors directly
    train_tensors = [t[train_idx] for t in dataset.tensors]
    val_tensors = [t[val_idx] for t in dataset.tensors]
    
    return TensorDataset(*train_tensors), TensorDataset(*val_tensors)