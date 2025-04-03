from flask.cli import F
from bayesian_torch.ao import nn
from bayesian_torch.layers import LSTMReparameterization
from bayesian_torch.layers import LinearReparameterization

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class BayesianRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BayesianRNN, self).__init__()
        self.lstm = LSTMReparameterization(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc = LinearReparameterization(
            in_features=hidden_size,
            out_features=output_size,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

    def forward(self, x):
        kl_sum = 0
        # Bayesian LSTM forward pass
        x, (hn, cn), kl = self.lstm(x)
        kl_sum += kl

        # Use the last hidden state for classification
        x = x[:, -1, :]  # Take the last time step

        # Bayesian Fully Connected Layer
        x, kl = self.fc(x)
        kl_sum += kl

        output = F.log_softmax(x, dim=1)
        return output, kl_sum