import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
import matplotlib


# To test:
# python -m pytest tests/test_all.py -v
#
# To get code coverage:
# python -m pytest --cov=flexAE tests/test_all.py
#
# To get code coverage html report to see which lines we missed:
# python -m pytest --cov=flexAE tests/test_all.py --cov-report=html





INIT_SETTINGS = [
    {
        "include_variational" : incl_var, 
        "include_scatter" :  incl_scat, 
        "include_error_head" :  incl_err, 
        "include_classifier" :  incl_class
     }
     for incl_var in [False, True]
     for incl_scat in [False, True]
     for incl_err in [False, True]
     for incl_class in [False, True]
]




### Initialize model tests
# Tests whether the model initializes with multiple different inputs.
class TestModelInitializes:

    def test_model_initializes_default_settings(self):
        from flexAE import flexAE

        model = flexAE()

        assert isinstance(model, flexAE)


    def test_model_initializes_simple(self):
        from flexAE import flexAE

        model = flexAE(
            input_dim = 4, 
            hidden_dim = [4,4], 
            latent_dim = 2, 
            classifier_dim=[4,4]
            )

        assert isinstance(model, flexAE)



    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_initializes(self, settings):
        from flexAE import flexAE

        model = flexAE(
            input_dim = 4, 
            hidden_dim = [4,4], 
            latent_dim = 2, 
            classifier_dim=[4,4],
            **settings
            )

        assert isinstance(model, flexAE)



### Encode method tests
# Tests whether the encode method runs with multiple different inputs and whether the outputs have correct dimensions.
class TestModelEncodes:

    def test_model_encode_simple(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)

        encode_output = model.encode(x)
        mu = encode_output['mu']

        assert mu.shape == (n_sample, latent_dim)


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_encode(self, settings):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )

        x = torch.randn(n_sample,input_dim)

        if settings['include_error_head']:
            x_err = torch.abs(torch.randn(n_sample,input_dim))
        else:
            x_err = None

        encode_output = model.encode(x, x_err)
        mu = encode_output['mu']

        assert mu.shape == (n_sample, latent_dim)

        if settings['include_variational']:
            logvar = encode_output['logvar']
            assert logvar.shape == (n_sample, latent_dim)



    @pytest.mark.parametrize(
            "n_sample, input_dim, latent_dim", 
            [
                (n_sample, input_dim, latent_dim)
                for n_sample in [5,10,15]
                for input_dim in [3,6,9]
                for latent_dim in [2,4,8]
            ],
    )
    def test_model_encode_multiple_sizes(self, n_sample, input_dim, latent_dim):
        from flexAE import flexAE


        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)

        encode_output = model.encode(x)
        mu = encode_output['mu']

        assert mu.shape == (n_sample, latent_dim)



### Decode method tests
# Tests whether the decode method runs with multiple different inputs and whether the outputs have correct dimensions.
class TestModelDecodes:

    def test_model_decode_simple(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        z = torch.randn(n_sample,latent_dim)
        decode_output = model.decode(z)
        assert decode_output.shape == (n_sample, input_dim)




    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_decode(self, settings):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )

        z = torch.randn(n_sample,latent_dim)
        decode_output = model.decode(z)
        assert decode_output.shape == (n_sample, input_dim)





    @pytest.mark.parametrize(
            "n_sample, input_dim, latent_dim", 
            [
                (n_sample, input_dim, latent_dim)
                for n_sample in [5,10,15]
                for input_dim in [3,6,9]
                for latent_dim in [2,4,8]
            ],
    )
    def test_model_decode_multiple_sizes(self, n_sample, input_dim, latent_dim):
        from flexAE import flexAE


        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        z = torch.randn(n_sample,latent_dim)

        decode_output = model.decode(z)
        assert decode_output.shape == (n_sample, input_dim)


### Classify method tests
# Tests whether the classify method runs with multiple different inputs and whether the outputs have correct dimensions.
class TestModelClassifies:

    def test_model_classifies_simple(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                include_classifier=True
                )

        z = torch.randn(n_sample,latent_dim)
        classify_output = model.classify(z)
        assert classify_output.shape == (n_sample, 1)


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_classifies(self, settings):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )

        if settings['include_classifier']:
            z = torch.randn(n_sample,latent_dim)
            classify_output = model.classify(z)
            assert classify_output.shape == (n_sample, 1)





    @pytest.mark.parametrize(
            "n_sample, input_dim, latent_dim", 
            [
                (n_sample, input_dim, latent_dim)
                for n_sample in [5,10,15]
                for input_dim in [3,6,9]
                for latent_dim in [2,4,8]
            ],
    )
    def test_model_classify_multiple_sizes(self, n_sample, input_dim, latent_dim):
        from flexAE import flexAE


        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                include_classifier=True
                )

        z = torch.randn(n_sample,latent_dim)

        classify_output = model.classify(z)
        assert classify_output.shape == (n_sample, 1)



### Forward pass tests
# Tests whether the forward pass runs with multiple different inputs and whether the outputs have correct dimensions.
class TestModelForwardPass:


    def test_model_forward_simple(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)
        forward_output = model(x)

        reconstructed_x = forward_output['reconstructed_x']
        mu = forward_output['mu']

        assert reconstructed_x.shape == (n_sample, input_dim)
        assert mu.shape == (n_sample, latent_dim)


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_forward(self, settings):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )

        x = torch.randn(n_sample,input_dim)

        if settings['include_error_head']:
            x_err = torch.abs(torch.randn(n_sample,input_dim))
        else:
            x_err = None


        forward_output = model(x, x_err)

        reconstructed_x = forward_output['reconstructed_x']
        mu = forward_output['mu']

        assert reconstructed_x.shape == (n_sample, input_dim)
        assert mu.shape == (n_sample, latent_dim)

        if settings['include_variational']:
            logvar = forward_output['logvar']
            assert logvar.shape == (n_sample, latent_dim)

        if settings['include_scatter']:
            scatter = forward_output['scatter']
            assert scatter.shape == (n_sample, input_dim)

        if settings['include_classifier']:
            classification_prediction = forward_output['classification_prediction']
            assert classification_prediction.shape == (n_sample, 1)



    @pytest.mark.parametrize(
            "n_sample, input_dim, latent_dim", 
            [
                (n_sample, input_dim, latent_dim)
                for n_sample in [5,10,15]
                for input_dim in [3,6,9]
                for latent_dim in [2,4,8]
            ],
    )
    def test_model_forward_multiple_sizes(self, n_sample, input_dim, latent_dim):
        from flexAE import flexAE

        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)
        forward_output = model(x)

        reconstructed_x = forward_output['reconstructed_x']
        mu = forward_output['mu']

        assert reconstructed_x.shape == (n_sample, input_dim)
        assert mu.shape == (n_sample, latent_dim)






class TestModelLossFunctions:

    def test_weighted_mse_loss(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        recon_x = torch.randn(n_sample,input_dim)
        scatter = torch.randn(n_sample,input_dim)

        loss = model.get_weighted_mse_loss(x,x_err,recon_x,scatter, penalty = True)
        assert loss.shape == ()


    def test_KLD_loss(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        mu = torch.randn(n_sample,latent_dim)
        logvar = torch.randn(n_sample,latent_dim)

        loss = model.get_KLD_loss(mu, logvar)
        assert loss.shape == ()



    def test_BCE_loss(self):
        from flexAE import flexAE

        n_sample = 10
        input_dim = 4
        latent_dim = 2
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        predictions = torch.tensor(np.random.rand(n_sample), dtype=torch.long).float().view(-1, 1) #Dimensions from [X] to [X,1]
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

        loss = model.get_BCE_loss(predictions, labels)
        assert loss.shape == ()



    def test_loss_function_simple(self):
        from flexAE import flexAE
        
        n_sample = 10
        input_dim = 4
        latent_dim = 2
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )

        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

        results = model(x, x_err)

        loss, _, _, _ = model.loss_function(x, x_err, results, labels)

        assert loss.shape == ()


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_loss_function(self, settings):
        from flexAE import flexAE
        
        n_sample = 10
        input_dim = 4
        latent_dim = 2
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )

        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

        results = model(x, x_err)
        loss, _, _, _ = model.loss_function(x, x_err, results, labels)

        assert loss.shape == ()



class TestModelTraining:

    def test_model_training_simple(self):
        '''
        Test whether the loss actually goes down during training.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2


        batch_size = 5
        epochs = 100
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )


        
        x = torch.randn(n_sample,input_dim)
        #x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                          epochs = epochs,
                          learning_rate = learning_rate, 
                          batch_size = batch_size,
                          validation_split = validation_split)

        loss_start = model.total_loss_per_epoch[0]
        loss_end = model.total_loss_per_epoch[-1]


        assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_model_training(self, settings):
        '''
        Test whether the loss actually goes down during training. Try all init settings.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        batch_size = 5
        epochs = 100
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )
        
        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        if settings['include_error_head'] and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, x_err, labels)

        if settings['include_error_head'] and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x, x_err)

        if settings['include_error_head'] == False and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, labels)

        if settings['include_error_head'] == False and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x)

        

        model.train_model(tensor_dataset, 
                          epochs = epochs,
                          learning_rate = learning_rate, 
                          batch_size = batch_size,
                          validation_split = validation_split)

        loss_start = model.total_loss_per_epoch[0]
        loss_end = model.total_loss_per_epoch[-1]

        assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"



    @pytest.mark.parametrize(
            "batch_size, epochs, learning_rate", 
            [
                (batch_size, epochs, learning_rate)
                for batch_size in [5,10]
                for epochs in [100,200]
                for learning_rate in [1e-2,1e-3]
            ],
    )
    def test_model_training_different_settings(self, batch_size, epochs, learning_rate):
        '''
        Test whether the loss actually goes down during training. Try a few different learning parameters.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )


        x = torch.randn(n_sample,input_dim)
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                          epochs = epochs,
                          learning_rate = learning_rate, 
                          batch_size = batch_size,
                          validation_split = validation_split)

        loss_start = model.total_loss_per_epoch[0]
        loss_end = model.total_loss_per_epoch[-1]


        assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"


class TestClassifierTraining:

    def test_classifier_training_simple(self):
        '''
        Test whether the loss actually goes down during training.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2


        batch_size = 5
        epochs = 100
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                include_classifier=True
                )


        
        x = torch.randn(n_sample,input_dim)
        #x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                                  epochs = epochs,
                                  learning_rate = learning_rate, 
                                  batch_size = batch_size,
                                  validation_split = validation_split)

        model.train_classifier(tensor_dataset, 
                          epochs = epochs,
                          learning_rate = learning_rate, 
                          batch_size = batch_size,
                          validation_split = validation_split)

        loss_start = model.total_loss_per_epoch[0]
        loss_end = model.total_loss_per_epoch[-1]


        assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_classifier_training(self, settings):
        '''
        Test whether the loss actually goes down during training. Try all init settings.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        batch_size = 5
        epochs = 100
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )
        
        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        if settings['include_error_head'] and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, x_err, labels)

        if settings['include_error_head'] and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x, x_err)

        if settings['include_error_head'] == False and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, labels)

        if settings['include_error_head'] == False and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x)


        if settings['include_classifier']:

            model.train_model(tensor_dataset, 
                                      epochs = epochs,
                                      learning_rate = learning_rate, 
                                      batch_size = batch_size,
                                      validation_split = validation_split)
            

            model.train_classifier(tensor_dataset, 
                            epochs = epochs,
                            learning_rate = learning_rate, 
                            batch_size = batch_size,
                            validation_split = validation_split)

            loss_start = model.total_loss_per_epoch[0]
            loss_end = model.total_loss_per_epoch[-1]

            assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"



    @pytest.mark.parametrize(
            "batch_size, epochs, learning_rate", 
            [
                (batch_size, epochs, learning_rate)
                for batch_size in [5,10]
                for epochs in [100,200]
                for learning_rate in [1e-2,1e-3]
            ],
    )
    def test_classifier_training_different_settings(self, batch_size, epochs, learning_rate):
        '''
        Test whether the loss actually goes down during training. Try a few different learning parameters.
        '''
        from flexAE import flexAE
                
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                include_classifier = True
                )


        x = torch.randn(n_sample,input_dim)
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                                  epochs = epochs,
                                  learning_rate = learning_rate, 
                                  batch_size = batch_size,
                                  validation_split = validation_split)
        

        model.train_classifier(tensor_dataset, 
                          epochs = epochs,
                          learning_rate = learning_rate, 
                          batch_size = batch_size,
                          validation_split = validation_split)

        loss_start = model.total_loss_per_epoch[0]
        loss_end = model.total_loss_per_epoch[-1]


        assert loss_end < loss_start,f"Loss at start: {loss_start}; loss at end: {loss_end}"



class TestPlottingMethods:
    """
    Not comparing/testing the actual plot pixel-by-pixel, but making sure the code actually runs.
    """
    @pytest.fixture(autouse=True)
    def cleanup_plots(self, monkeypatch):
        """Automatically close all figures after each test to prevent memory leaks."""

        matplotlib.use('Agg')
        monkeypatch.setattr(plt, "show", lambda: None)

        yield

        plt.close('all')

            

    def test_plotting_methods_simple(self):

        from flexAE import flexAE
                        
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        batch_size = 5
        epochs = 10
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4]
                )


        x = torch.randn(n_sample,input_dim)
        #x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                            epochs = epochs,
                            learning_rate = learning_rate, 
                            batch_size = batch_size,
                            validation_split = validation_split)

        model.plot_loss_per_epoch()
        model.plot_loss_stackplot()
        model.plot_learningrate_per_epoch()
        model.plot_feature_residuals_per_epoch()
        model.plot_feature_residuals_per_epoch_v2()
        model.plot_feature_residuals()
        if model.include_scatter:
            model.plot_feature_scatter_per_epoch()
            model.plot_feature_scatter_per_epoch_v2()
            model.plot_feature_scatter()
        model.compare_reconstructed_features(tensor_dataset)
        model.plot_mu_per_epoch()
        if model.include_variational:
            model.plot_logvar_per_epoch()
        model.parameter_summary(print_summary=False)


    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_plotting_methods(self, settings):
        from flexAE import flexAE
                        
        n_sample = 10
        input_dim = 4
        latent_dim = 2

        batch_size = 5
        epochs = 10
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                **settings
                )


        x = torch.randn(n_sample,input_dim)
        x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)


        if settings['include_error_head'] and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, x_err, labels)

        if settings['include_error_head'] and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x, x_err)

        if settings['include_error_head'] == False and settings['include_classifier']:
            tensor_dataset = TensorDataset(x, labels)

        if settings['include_error_head'] == False and settings['include_classifier'] == False:
            tensor_dataset = TensorDataset(x)


        model.train_model(tensor_dataset, 
                            epochs = epochs,
                            learning_rate = learning_rate, 
                            batch_size = batch_size,
                            validation_split = validation_split)

        model.plot_loss_per_epoch()
        model.plot_loss_stackplot()
        model.plot_learningrate_per_epoch()
        model.plot_feature_residuals_per_epoch()
        model.plot_feature_residuals_per_epoch_v2()
        model.plot_feature_residuals()
        if model.include_scatter:
            model.plot_feature_scatter_per_epoch()
            model.plot_feature_scatter_per_epoch_v2()
            model.plot_feature_scatter()
        model.compare_reconstructed_features(tensor_dataset)
        model.plot_mu_per_epoch()
        if model.include_variational:
            model.plot_logvar_per_epoch()
        model.parameter_summary(print_summary=False)



class TestLatentScore:

    def test_latent_score_simple(self):
        from flexAE import flexAE
                                
        n_sample = 40
        input_dim = 4
        latent_dim = 2

        batch_size = 5
        epochs = 10
        learning_rate = 1e-2
        validation_split = 0.1
        
        model = flexAE(
                input_dim = input_dim, 
                hidden_dim = [4,4], 
                latent_dim = latent_dim, 
                classifier_dim=[4,4],
                #**settings
                )

        x = torch.randn(n_sample,input_dim)
        #x_err = torch.abs(torch.randn(n_sample,input_dim))
        labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

        tensor_dataset = TensorDataset(x, labels)

        model.train_model(tensor_dataset, 
                            epochs = epochs,
                            learning_rate = learning_rate, 
                            batch_size = batch_size,
                            validation_split = validation_split)
        
        model.build_latent_map(tensor_dataset)
        dist_score = model.get_latent_distance_score(x)

        assert dist_score.shape == (n_sample,)

    @pytest.mark.parametrize("settings", INIT_SETTINGS)
    def test_latent_score(self, settings):
            from flexAE import flexAE
                                    
            n_sample = 40
            input_dim = 4
            latent_dim = 2

            batch_size = 5
            epochs = 10
            learning_rate = 1e-2
            validation_split = 0.1
            
            model = flexAE(
                    input_dim = input_dim, 
                    hidden_dim = [4,4], 
                    latent_dim = latent_dim, 
                    classifier_dim=[4,4],
                    **settings
                    )

            x = torch.randn(n_sample,input_dim)
            x_err = torch.abs(torch.randn(n_sample,input_dim))
            labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

            if settings['include_error_head'] and settings['include_classifier']:
                tensor_dataset = TensorDataset(x, x_err, labels)

            if settings['include_error_head'] and settings['include_classifier'] == False:
                tensor_dataset = TensorDataset(x, x_err)

            if settings['include_error_head'] == False and settings['include_classifier']:
                tensor_dataset = TensorDataset(x, labels)

            if settings['include_error_head'] == False and settings['include_classifier'] == False:
                tensor_dataset = TensorDataset(x)

            model.train_model(tensor_dataset, 
                                epochs = epochs,
                                learning_rate = learning_rate, 
                                batch_size = batch_size,
                                validation_split = validation_split)
            
            model.build_latent_map(tensor_dataset)
            dist_score = model.get_latent_distance_score(x)

            assert dist_score.shape == (n_sample,)




def test_split_tensordataset():
    from flexAE import split_tensordataset

    n_sample = 40
    input_dim = 4

    x = torch.randn(n_sample,input_dim)
    x_err = torch.abs(torch.randn(n_sample,input_dim))
    labels = torch.tensor(np.round(np.random.rand(n_sample)), dtype=torch.long)

    tensor_dataset = TensorDataset(x, x_err, labels)

    split_fraction = 0.8
    train_dataset, val_dataset = split_tensordataset(tensor_dataset, train_fraction=split_fraction)

    assert np.allclose(len(train_dataset), np.round(split_fraction * n_sample))
    assert np.allclose(len(val_dataset), np.round((1-split_fraction) * n_sample))