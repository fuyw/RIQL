import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
from sklearn.model_selection import KFold, train_test_split


class MLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.output_dim, name="probe_fc")(x)
        return x


class ProbeTrainer:

    def __init__(self,
                 input_dim,
                 output_dim,
                 batch_size=256,
                 lr=3e-4,
                 epochs=200):
        self.mlp = MLP(output_dim)
        self.epochs = epochs
        rng = jax.random.PRNGKey(0)
        dummy_inputs = jnp.ones(input_dim)
        params = self.mlp.init(rng, dummy_inputs)["params"]
        self.state = train_state.TrainState.create(
            apply_fn=self.mlp.apply,
            params=params,
            tx=optax.adam(learning_rate=lr))
        self.batch_size = batch_size

    def train(self, X, Y):

        @jax.jit
        def train_step(model_state: train_state.TrainState,
                       batch_X: jnp.ndarray, batch_Y: jnp.ndarray):

            def loss_fn(params, x, y):
                output = self.mlp.apply({"params": params}, x)
                loss = (output - y)**2
                return loss.mean()

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(model_state.params, batch_X, batch_Y)
            new_model_state = model_state.apply_gradients(grads=grads)
            return new_model_state, loss

        @jax.jit
        def eval_step(params: FrozenDict, batch_X: jnp.ndarray,
                      batch_Y: jnp.ndarray):

            def loss_fn(x, y):
                output = self.mlp.apply({"params": params}, x)
                loss = (output - y)**2
                return loss.mean()

            loss = loss_fn(batch_X, batch_Y)
            return loss

        kf = KFold(n_splits=5)
        kf_losses = []
        for trainval_idx, test_idx in kf.split(X):
            test_X = X[test_idx]
            train_idx, valid_idx = train_test_split(trainval_idx,
                                                    test_size=0.1)
            train_X, valid_X, test_X = X[train_idx], X[valid_idx], X[test_idx]
            train_Y, valid_Y, test_Y = Y[train_idx], Y[valid_idx], Y[test_idx]

            batch_num = int(np.ceil(len(train_idx) / self.batch_size))
            min_valid_loss = np.inf
            optimal_params = None
            patience = 0
            for epoch in range(self.epochs):
                epoch_loss = 0
                for i in range(batch_num):
                    batch_X = train_X[i * self.batch_size:(i + 1) *
                                      self.batch_size]
                    batch_Y = train_Y[i * self.batch_size:(i + 1) *
                                      self.batch_size]
                    self.state, batch_loss = train_step(
                        self.state, batch_X, batch_Y)
                    epoch_loss += batch_loss
                valid_loss = eval_step(self.state.params, valid_X, valid_Y)
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    patience = 0
                    optimal_params = self.state.params
                else:
                    patience += 1
                # print(f'# Epoch {epoch}: train_loss: {epoch_loss/batch_num:6f}, valid_loss: {valid_loss:.6f}')
                if patience == 10:
                    # print(f'Early break at epoch {epoch}.')
                    break

            # test
            test_loss = eval_step(optimal_params, test_X, test_Y)
            kf_losses.append(test_loss.item())
        return kf_losses
