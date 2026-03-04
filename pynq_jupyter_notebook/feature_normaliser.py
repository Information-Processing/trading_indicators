import numpy as np


class FeatureNormaliser:
      def __init__(self, num_features, quant_bits=10):
          self.num_features = num_features
          self.quant_max = (1 << (quant_bits - 1)) - 1
          self.n = 0
          self._mean = np.zeros(num_features)
          self._m2 = np.zeros(num_features)

      def _update_stats(self, batch):
          batch_n = batch.shape[0]
          if batch_n == 0:
              return
          batch_mean = batch.mean(axis=0)
          batch_m2 = batch.var(axis=0, ddof=0) * batch_n
          if self.n == 0:
              self._mean = batch_mean
              self._m2 = batch_m2
          else:
              total_n = self.n + batch_n
              delta = batch_mean - self._mean
              self._mean += delta * (batch_n / total_n)
              self._m2 += batch_m2 + delta ** 2 * (self.n * batch_n / total_n)
          self.n += batch_n

      @property
      def std(self):
          if self.n < 2:
              return np.ones(self.num_features)
          s = np.sqrt(self._m2 / (self.n - 1))
          s[s < 1e-10] = 1.0
          return s

      def normalise_and_quantise(self, samples):
          self._update_stats(samples)
          z = (samples - self._mean) / self.std
          z = np.clip(z, -3.0, 3.0)
          return np.rint(z * (self.quant_max / 3.0)).astype(np.int32)

      def denormalise_weights(self, weights_norm):
          """
          Convert weights from normalized/quantized space back to original units.
          weights_norm: (D,) or (D, 1) where D = num_features (13).
          First 12 entries are feature weights, last entry is the bias weight.

          In normalized space:  ỹ = Σ(w̃_i · x̃_i) + w̃_bias
          In original space:    y = Σ(w_i · x_i) + b
          where w_i = w̃_i · σ_y / σ_i
                b   = w̃_bias · σ_y / s + μ_y − Σ(w_i · μ_i)
          """
          orig_shape = weights_norm.shape
          w = weights_norm.flatten()

          feat_std = self.std[:-1]
          target_std = self.std[-1]
          feat_mean = self._mean[:-1]
          target_mean = self._mean[-1]
          s = self.quant_max / 3.0

          w_feat = w[:-1] * (target_std / feat_std)
          w_bias = w[-1] * (target_std / s) + target_mean - np.sum(w_feat * feat_mean)

          return np.concatenate([w_feat, [w_bias]]).reshape(orig_shape)
