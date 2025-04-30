import math
import torch
import torch.nn as nn


class BatchNormalNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 order=2,
                 alpha=1.0,
                 iterations=1,
                 standardize=True,
                 affine=True,
                 track_running_stats=True,
                 momentum=0.1,
                 noise_train=0.4,
                 eps=1e-05,
                 device=None):
        super(BatchNormalNorm1d, self).__init__()

        self.num_features = num_features
        self.standardize = standardize
        self.affine = affine
        self.order = order
        self.iterations = iterations
        self.alpha = alpha
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.num_batches_tracked = 0

        # running averages of transformation parameters
        self.register_buffer('running_lmbda', torch.ones(self.num_features).to(device))
        self.register_buffer('running_mean', torch.zeros(self.num_features).to(device))
        self.register_buffer('running_var', torch.ones(self.num_features).to(device))
        self.register_buffer('running_norm', math.sqrt(2. / math.pi) * torch.ones(self.num_features).to(device))

        # cached variables stored here
        self._init_cache()

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        dims = x.dim()
        if dims != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(dims))

    def _init_cache(self):
        self.cache = {'lmbda_estimate': None}

    def _reset_cache(self):
        self.cache.clear()
        self._init_cache()

    def forward(self, x):
        self._reset_cache()
        self._check_input_dim(x)

        exp_avg_factor = 0.
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exp_avg_factor = 1. / self.num_batches_tracked
                else: # use exponential moving average
                    exp_avg_factor = self.momentum

        if self.training:
            n, c = x.shape
            numel = n

            if self.standardize:
                mean = torch.mean(x, dim=(0,))
                var = torch.var(x, dim=(0,), correction=0)
                x = self._standardize(x, mean, var)

            x_sign = torch.sign(x)
            x_sign[x_sign == 0] = 1.
            x_abs = torch.abs(x)

            lmbda = self._estimate(x, x_sign, x_abs, order=self.order)
            self.cache['lmbda_estimate'] = lmbda.detach().clone()
            x = self._transform(x_sign, x_abs, lmbda)

            with torch.no_grad():
                mean_ = torch.mean(x, dim=(0,), keepdim=True)
                norm_ = torch.mean(torch.abs(x - mean_), dim=(0,))
                norm_cast = norm_[None, :]

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_lmbda = exp_avg_factor * lmbda + (1. - exp_avg_factor) * self.running_lmbda
                    self.running_mean = exp_avg_factor * mean + (1. - exp_avg_factor) * self.running_mean
                    self.running_var = exp_avg_factor * var * (numel) / (numel - 1.) + (1. - exp_avg_factor) * self.running_var
                    self.running_norm = exp_avg_factor * norm_ * (numel) / (numel - 1) + (1. - exp_avg_factor) * self.running_norm

            x = scaled_additive_normal_noise(x, norm_cast, mean=0., std=self.noise_train)

        else:
            if self.standardize:
                x = self._standardize(x, self.running_mean, self.running_var)

            x_sign = torch.sign(x)
            x_sign[x_sign == 0] = 1.
            x_abs = torch.abs(x)

            x = self._transform(x_sign, x_abs, self.running_lmbda)
            norm_cast = self.running_norm[None, :]

        if self.affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _estimate(self, x, x_sign, x_abs, order=2):
        c = x_abs.shape[1]

        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones(c).to(x_abs.device) - self.alpha * d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones(c).to(x_abs.device) - self.alpha * d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=(0,))
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=(0,))
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=(0,))
            d_sub_avg = d - dmean[None, :]
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=(0,))
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)[None, :]
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean[None, :]) / torch.sqrt(var[None, :] + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain[None, :] + shift[None, :]


class BatchNormalNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 order=2,
                 alpha=1.0,
                 iterations=1,
                 standardize=True,
                 affine=True,
                 track_running_stats=True,
                 momentum=0.1,
                 noise_train=0.4,
                 eps=1e-05,
                 device=None):
        super(BatchNormalNorm2d, self).__init__()

        self.num_features = num_features
        self.standardize = standardize
        self.affine = affine
        self.order = order
        self.iterations = iterations
        self.alpha = alpha
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.num_batches_tracked = 0

        # running averages of transformation parameters
        self.register_buffer('running_lmbda', torch.ones(self.num_features).to(device))
        self.register_buffer('running_mean', torch.zeros(self.num_features).to(device))
        self.register_buffer('running_var', torch.ones(self.num_features).to(device))
        self.register_buffer('running_norm', math.sqrt(2. / math.pi) * torch.ones(self.num_features).to(device))

        # cached variables stored here
        self._init_cache()

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        dims = x.dim()
        if dims != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(dims))

    def _init_cache(self):
        self.cache = {'lmbda_estimate': None}

    def _reset_cache(self):
        self.cache.clear()
        self._init_cache()

    def forward(self, x):
        self._reset_cache()
        self._check_input_dim(x)

        exp_avg_factor = 0.
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exp_avg_factor = 1. / self.num_batches_tracked
                else: # use exponential moving average
                    exp_avg_factor = self.momentum

        if self.training:
            n, c, h, w = x.shape
            numel = n * h * w

            if self.standardize:
                mean = torch.mean(x, dim=(0, 2, 3))
                var = torch.var(x, dim=(0, 2, 3), correction=0)
                x = self._standardize(x, mean, var)

            x_sign = torch.sign(x)
            x_sign[x_sign == 0] = 1.
            x_abs = torch.abs(x)

            lmbda = self._estimate(x, x_sign, x_abs, order=self.order)
            self.cache['lmbda_estimate'] = lmbda.detach().clone()
            x = self._transform(x_sign, x_abs, lmbda)

            with torch.no_grad():
                mean_ = torch.mean(x, dim=(0, 2, 3), keepdim=True)
                norm_ = torch.mean(torch.abs(x - mean_), dim=(0, 2, 3))
                norm_cast = norm_[None, :, None, None]

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_lmbda = exp_avg_factor * lmbda + (1. - exp_avg_factor) * self.running_lmbda
                    self.running_mean = exp_avg_factor * mean + (1. - exp_avg_factor) * self.running_mean
                    self.running_var = exp_avg_factor * var * (numel) / (numel - 1.) + (1. - exp_avg_factor) * self.running_var
                    self.running_norm = exp_avg_factor * norm_ * (numel) / (numel - 1) + (1. - exp_avg_factor) * self.running_norm

            x = scaled_additive_normal_noise(x, norm_cast, mean=0., std=self.noise_train)

        else:
            if self.standardize:
                x = self._standardize(x, self.running_mean, self.running_var)

            x_sign = torch.sign(x)
            x_sign[x_sign == 0] = 1.
            x_abs = torch.abs(x)

            x = self._transform(x_sign, x_abs, self.running_lmbda)
            norm_cast = self.running_norm[None, :, None, None]

        if self.affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _estimate(self, x, x_sign, x_abs, order=2):
        c = x_abs.shape[1]

        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones(c).to(x_abs.device) - self.alpha * d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones(c).to(x_abs.device) - self.alpha * d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=(0, 2, 3))
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=(0, 2, 3))
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=(0, 2, 3))
            d_sub_avg = d - dmean[None, :, None, None]
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=(0, 2, 3))
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)[None, :, None, None]
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain[None, :, None, None] + shift[None, :, None, None]


class LayerNormalNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 noise_train=1.0,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(LayerNormalNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, (list, tuple)):
            normalized_shape = torch.Size(normalized_shape)
        self.normalized_shape = normalized_shape
        self._dims = [-(i+1) for i in range(len(self.normalized_shape))]
        self.elementwise_affine = elementwise_affine
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        # cached variables stored here
        self._init_cache()

        if self.elementwise_affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.normalized_shape).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.normalized_shape).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        if self.normalized_shape != x.shape[-len(self.normalized_shape):]:
            raise ValueError('input shape {} inconsistent with normalized_shape {}'.format(x.shape, self.normalized_shape))

    def _init_cache(self):
        self.cache = {'lmbda_estimate': None}

    def _reset_cache(self):
        self.cache.clear()
        self._init_cache()

    def forward(self, x):
        self._reset_cache()
        self._check_input_dim(x)

        mean = torch.mean(x, dim=self._dims, keepdim=True)
        var = torch.var(x, correction=0, dim=self._dims, keepdim=True)
        x = self._standardize(x, mean, var)

        x_sign = torch.sign(x)
        x_sign[x_sign == 0] = 1.
        x_abs = torch.abs(x)

        lmbda = self._estimate(x, x_sign, x_abs)
        self.cache['lmbda_estimate'] = lmbda.detach().clone()
        x = self._transform(x_sign, x_abs, lmbda)

        with torch.no_grad():
            mean_ = torch.mean(x, dim=self._dims, keepdim=True)
            norm_cast = torch.mean(torch.abs(x - mean_), dim=self._dims, keepdim=True)
        if self.training:
            x = scaled_additive_normal_noise(x, norm_cast, mean=0., std=self.noise_train)

        if self.elementwise_affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _estimate(self, x, x_sign, x_abs, order=2):
        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones(size=self.normalized_shape).to(x_abs.device) - d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones(size=self.normalized_shape).to(x_abs.device) - d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=self._dims, keepdim=True)
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=self._dims, keepdim=True)
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=self._dims, keepdim=True)
            d_sub_avg = d - dmean
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=self._dims, keepdim=True)
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain + shift


class BatchNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 standardize=True,
                 affine=True,
                 track_running_stats=True,
                 momentum=0.1,
                 eps=1e-05,
                 device=None,
                 *args, **kwargs):
        super(BatchNorm1d, self).__init__()

        self.num_features = num_features
        self.standardize = standardize
        self.affine = affine
        self.eps = eps

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.num_batches_tracked = 0

        # running averages of transformation parameters
        self.register_buffer('running_mean', torch.zeros(self.num_features).to(device))
        self.register_buffer('running_var', torch.ones(self.num_features).to(device))

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        dims = x.dim()
        if dims != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(dims))

    def forward(self, x):
        self._check_input_dim(x)

        exp_avg_factor = 0.
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exp_avg_factor = 1. / self.num_batches_tracked
                else: # use exponential moving average
                    exp_avg_factor = self.momentum

        if self.training:
            n, c = x.shape
            numel = n

            if self.standardize:
                mean = torch.mean(x, dim=(0,))
                var = torch.var(x, dim=(0,), correction=0)
                x = self._standardize(x, mean, var)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exp_avg_factor * mean + (1. - exp_avg_factor) * self.running_mean
                    self.running_var = exp_avg_factor * var * (numel) / (numel - 1.) + (1. - exp_avg_factor) * self.running_var
        else:
            if self.standardize:
                x = self._standardize(x, self.running_mean, self.running_var)

        if self.affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _standardize(self, x, mean, var):
        return (x - mean[None, :]) / torch.sqrt(var[None, :] + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain[None, :] + shift[None, :]


class BatchNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 standardize=True,
                 affine=True,
                 track_running_stats=True,
                 momentum=0.1,
                 eps=1e-05,
                 device=None,
                 *args, **kwargs):
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.standardize = standardize
        self.affine = affine
        self.eps = eps

        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.num_batches_tracked = 0

        # running averages of transformation parameters
        self.register_buffer('running_mean', torch.zeros(self.num_features).to(device))
        self.register_buffer('running_var', torch.ones(self.num_features).to(device))

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        dims = x.dim()
        if dims != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(dims))

    def forward(self, x):
        self._check_input_dim(x)

        exp_avg_factor = 0.
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exp_avg_factor = 1. / self.num_batches_tracked
                else: # use exponential moving average
                    exp_avg_factor = self.momentum

        if self.training:
            n, c, h, w = x.shape
            numel = n * h * w

            if self.standardize:
                mean = torch.mean(x, dim=(0, 2, 3))
                var = torch.var(x, dim=(0, 2, 3), correction=0)
                x = self._standardize(x, mean, var)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exp_avg_factor * mean + (1. - exp_avg_factor) * self.running_mean
                    self.running_var = exp_avg_factor * var * (numel) / (numel - 1.) + (1. - exp_avg_factor) * self.running_var
        else:
            if self.standardize:
                x = self._standardize(x, self.running_mean, self.running_var)

        if self.affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _standardize(self, x, mean, var):
        return (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain[None, :, None, None] + shift[None, :, None, None]


class LayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(LayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, (list, tuple)):
            normalized_shape = torch.Size(normalized_shape)
        self.normalized_shape = normalized_shape
        self._dims = [-(i+1) for i in range(len(self.normalized_shape))]
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if self.elementwise_affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.normalized_shape).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.normalized_shape).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def _check_input_dim(self, x):
        if self.normalized_shape != x.shape[-len(self.normalized_shape):]:
            raise ValueError('input shape {} inconsistent with normalized_shape {}'.format(x.shape, self.normalized_shape))

    def forward(self, x):
        self._check_input_dim(x)

        mean = torch.mean(x, dim=self._dims, keepdim=True)
        var = torch.var(x, correction=0, dim=self._dims, keepdim=True)
        x = self._standardize(x, mean, var)

        if self.elementwise_affine:
            x = self._destandardize(x, self.bias, self.weight)
        return x

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain + shift


class GroupNormalNorm(nn.Module):
    def __init__(self,
                 num_channels,
                 num_groups=32,
                 eps=1e-05,
                 affine=True,
                 noise_train=0.4,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(GroupNormalNorm, self).__init__()

        assert num_channels % num_groups == 0, 'Number of channels should be evenly divisible by the number of groups'

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.affine = affine
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_channels).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_channels).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.num_channels == x_shape[1]

        x = x.reshape(batch_size, self.num_groups, -1)
        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, correction=0, dim=[-1], keepdim=True)

        x_norm = self._standardize(x, mean, var)
        x_norm = x_norm.reshape(batch_size, self.num_groups, -1)

        x_sign = torch.sign(x_norm)
        x_sign[x_sign == 0] = 1.
        x_abs = torch.abs(x_norm)

        lmbda = self._estimate(x_norm, x_sign, x_abs)
        x_norm = self._transform(x_sign, x_abs, lmbda)

        with torch.no_grad():
            mean_ = torch.mean(x_norm, dim=[-1], keepdim=True)
            norm_cast = torch.mean(torch.abs(x_norm - mean_), dim=[-1], keepdim=True)
        if self.training:
            x_norm = scaled_additive_normal_noise(x_norm, norm_cast, mean=0., std=self.noise_train)

        if self.affine:
            x_norm = x_norm.reshape(batch_size, self.num_channels, -1)
            x_norm = self._destandardize(x_norm, self.bias, self.weight)

        return x_norm.reshape(x_shape)

    def _estimate(self, x, x_sign, x_abs, order=2):
        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones((x.shape[0], self.num_groups, 1,)).to(x_abs.device) - d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones((x.shape[0], self.num_groups, 1,)).to(x_abs.device) - d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=[-1], keepdim=True)
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=[-1], keepdim=True)
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=[-1], keepdim=True)
            d_sub_avg = d - dmean
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=[-1], keepdim=True)
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain.reshape(1, -1, 1) + shift.reshape(1, -1, 1)


class InstanceNormalNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-05,
                 noise_train=0.4,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(InstanceNormalNorm2d, self).__init__()

        self.num_features = num_features
        self.affine = affine
        self.noise_train = noise_train
        self.eps = eps
        self.eps_sqrt = math.sqrt(eps)

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.num_features == x.shape[1]

        x = x.reshape(batch_size, self.num_features, -1)
        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, correction=0, dim=[-1], keepdim=True)
        x_norm = self._standardize(x, mean, var)
        x_norm = x_norm.reshape(batch_size, self.num_features, -1)

        x_sign = torch.sign(x_norm)
        x_sign[x_sign == 0] = 1.
        x_abs = torch.abs(x_norm)

        lmbda = self._estimate(x_norm, x_sign, x_abs)
        x_norm = self._transform(x_sign, x_abs, lmbda)

        with torch.no_grad():
            mean_ = torch.mean(x_norm, dim=[-1], keepdim=True)
            norm_cast = torch.mean(torch.abs(x_norm - mean_), dim=[-1], keepdim=True)
        if self.training:
            x_norm = scaled_additive_normal_noise(x_norm, norm_cast, mean=0., std=self.noise_train)

        if self.affine:
            x_norm = self._destandardize(x_norm, self.bias, self.weight)

        return x_norm.reshape(x_shape)

    def _estimate(self, x, x_sign, x_abs, order=2):
        d1lmbda, d2lmbda = self._compute_grads(x, x_sign, x_abs, order=order)

        if order == 1:
            lmbda = torch.ones((x.shape[0], self.num_features, 1,)).to(x_abs.device) - d1lmbda # gradient descent update
        elif order == 2:
            lmbda = torch.ones((x.shape[0], self.num_features, 1,)).to(x_abs.device) - d1lmbda / (d2lmbda + self.eps_sqrt) # newton-raphson update
        return lmbda

    def _compute_grads(self, x, x_sign, x_abs, order=2):
        x_abs_log1p = torch.log1p(x_abs)
        x_masked = x_sign * x_abs_log1p

        s1 = torch.mean(x_masked, dim=[-1], keepdim=True)
        d = (1. + x_abs) * x_abs_log1p - x_abs
        t1 = x * d
        dvar = 2. * torch.mean(t1, dim=[-1], keepdim=True)
        g1 = 0.5 * dvar - s1

        if order == 2:
            dmean = torch.mean(d, dim=[-1], keepdim=True)
            d_sub_avg = d - dmean
            d_sub_avg_square = torch.square(d_sub_avg)

            x_abs_log1p_square = torch.square(x_abs_log1p)
            p1 = (1. + x_abs) * x_abs_log1p_square - 2. * d
            d2 = x_sign * p1
            t2 = x * d2 + d_sub_avg_square
            d2var = 2. * torch.mean(t2, dim=[-1], keepdim=True)
            dvar_square = torch.square(dvar)
            t3_1 = -0.5 * dvar_square
            t3_2 = 0.5 * d2var
            g2 = t3_1 + t3_2
            return g1, g2
        return g1, None

    def _transform(self, x_sign, x_abs, lmbda):
        eta = 1. + x_sign * (lmbda - 1.)
        with torch.no_grad():
            eta_sign = torch.sign(eta)
            eta_sign[eta_sign == 0] = 1.

        p1 = x_sign / (eta + eta_sign * self.eps_sqrt)
        p2 = torch.pow(1. + x_abs, eta + eta_sign * self.eps_sqrt) - 1.
        x_tr1 = p1 * p2

        x_tr2 = x_sign * torch.log1p(x_abs)

        with torch.no_grad():
            mask = (torch.abs(eta) <= self.eps_sqrt)
        x_tr = (mask == 0).to(torch.float32) * x_tr1 + (mask == 1).to(torch.float32) * x_tr2
        return x_tr

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain.reshape(1, -1, 1) + shift.reshape(1, -1, 1)


class GroupNorm(nn.Module):
    def __init__(self,
                 num_channels,
                 num_groups=32,
                 eps=1e-05,
                 affine=True,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(GroupNorm, self).__init__()

        assert num_channels % num_groups == 0, 'Number of channels should be evenly divisible by the number of groups'

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_channels).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_channels).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.num_channels == x_shape[1]

        x = x.reshape(batch_size, self.num_groups, -1)
        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, correction=0, dim=[-1], keepdim=True)

        x_norm = self._standardize(x, mean, var)
        x_norm = x_norm.reshape(batch_size, self.num_groups, -1)

        if self.affine:
            x_norm = x_norm.reshape(batch_size, self.num_channels, -1)
            x_norm = self._destandardize(x_norm, self.bias, self.weight)

        return x_norm.reshape(x_shape)

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain.reshape(1, -1, 1) + shift.reshape(1, -1, 1)


class InstanceNorm2d(nn.Module):
    def __init__(self,
                 num_features,
                 affine=True,
                 eps=1e-05,
                 device=None,
                 dtype=None,
                 *args, **kwargs):
        super(InstanceNorm2d, self).__init__()

        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.bias = nn.parameter.Parameter(torch.zeros(self.num_features).to(device))
            self.weight = nn.parameter.Parameter(torch.ones(self.num_features).to(device))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)

    def forward(self, x):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.num_features == x.shape[1]

        x = x.reshape(batch_size, self.num_features, -1)
        mean = torch.mean(x, dim=[-1], keepdim=True)
        var = torch.var(x, correction=0, dim=[-1], keepdim=True)

        x_norm = self._standardize(x, mean, var)
        x_norm = x_norm.reshape(batch_size, self.num_features, -1)

        if self.affine:
            x_norm = self._destandardize(x_norm, self.bias, self.weight)

        return x_norm.reshape(x_shape)

    def _standardize(self, x, mean, var):
        return (x - mean) / torch.sqrt(var + self.eps)

    def _destandardize(self, x, shift, gain):
        return x * gain.reshape(1, -1, 1) + shift.reshape(1, -1, 1)


def scaled_additive_normal_noise(x, scale, mean=0., std=1.):
    x = x + (torch.randn_like(x) * scale * std + mean)
    return x


class Standardize(nn.Module):
    def __init__(self,
                 mean,
                 std,
                 device=None):
        super(Standardize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).to(device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return self._standardize(x, self.mean, self.std)

    @staticmethod
    def _standardize(x, mean, std):
        ndim = len(x.shape)
        if ndim == 4:
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
        elif ndim == 3:
            mean = mean[None, :, None]
            std = std[None, :, None]
        elif ndim == 2:
            mean = mean[None, :]
            std = std[None, :]
        elif ndim >= 5:
            raise RuntimeError('input dimension is larger than 4')
        return x.sub(mean).div(std)
