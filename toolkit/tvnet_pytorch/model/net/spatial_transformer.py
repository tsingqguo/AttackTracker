# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pysot.core.config import cfg

class spatial_transformer(nn.Module):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_ and edited by Gasoon Jia for Pytorch.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, num_channels, height, width].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/LijieFan/tvnet/blob/master/spatial_transformer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def __init__(self, name='SpatialTransformer'):
        super(spatial_transformer, self).__init__()
        self.name = name

    
    def forward(self, U, theta, out_size):
        output = self._transform(theta, U, out_size)
        return output
    
    def _transform(self, theta, input_dim, out_size):
        num_batch = input_dim.size(0)
        num_channels = input_dim.size(1)
        height = input_dim.size(2)
        width = input_dim.size(3)

        theta = theta.float()
        
        out_height = out_size[0]
        out_width = out_size[1]
        grid = self._meshgrid(out_height, out_width)
        grid = grid[None, ...] # why do this since view to just one dimention below immediately
        grid = grid.view(-1)
        grid = grid.repeat(num_batch)
        grid = grid.view(num_batch, 2, -1).float() #TODO: CHECK SHAPE
        T_g = theta + Variable(grid)

        x_s = T_g.data[:, 0: 1, :]
        y_s = T_g.data[:, 1: 2, :]
        x_s_flat = x_s.view(-1)
        y_s_flat = y_s.view(-1)

        input_transformed = self._interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        input_transformed = input_transformed.view(num_batch, out_height, out_width, num_channels).permute(0, 3, 1, 2)

        output = Variable(input_transformed)
        return output
    
    def _meshgrid(self, height, width):

        x_t = torch.matmul(torch.ones(height, 1), 
                           torch.transpose(torch.linspace(-1.0, 1.0, width)[:, None], 1, 0))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height)[:, None], torch.ones(1, width))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        if cfg.CUDA:
            grid = torch.cat([x_t_flat, y_t_flat], dim=0).cuda()
        else:
            grid = torch.cat([x_t_flat, y_t_flat], dim=0)
        return grid
    
    def _interpolate(self, im, x, y, out_size):
        num_batch = im.size(0)
        channels = im.size(1)
        height = im.size(2)
        width = im.size(3)

        x = x.float()
        y = y.float()
        height_f = float(height)
        width_f = float(width)

        out_height = out_size[0]
        out_width = out_size[1]
        zero = 0
        max_y = int(height - 1)
        max_x = int(width - 1)

        x = (x + 1) * (width_f - 1) / 2.0
        y = (y + 1) * (height_f - 1) / 2.0

        x0 = x.floor().int()
        y0 = y.floor().int()
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = x0.clamp(zero, max_x - 1)
        x1 = x1.clamp(zero, max_x)
        y0 = y0.clamp(zero, max_y - 1)
        y1 = y1.clamp(zero, max_y)
        dim2 = width
        dim1 = width * height

        base = self._repeat(torch.arange(0, num_batch).int() * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im = im.permute(0, 2, 3, 1)
        im_flat = im.view(-1, channels).float()
        Ia = torch.index_select(im_flat, dim=0, index=Variable(idx_a.long()))
        Ib = torch.index_select(im_flat, dim=0, index=Variable(idx_b.long()))
        Ic = torch.index_select(im_flat, dim=0, index=Variable(idx_c.long()))
        Id = torch.index_select(im_flat, dim=0, index=Variable(idx_d.long()))

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y))[:, None]
        wb = ((x1_f - x) * (y - y0_f))[:, None]
        wc = ((x - x0_f) * (y1_f - y))[:, None]
        wd = ((x - x0_f) * (y - y0_f))[:, None]
        
        output = wa * Ia.data + wb * Ib.data + wc * Ic.data + wd * Id.data
        return output
    
    def _repeat(self, x, n_repeats):
        rep = torch.ones([1, n_repeats]).int()
        # There's some differnent between my implementation and original's
        # If something wrong, should change it back to original type
        if cfg.CUDA:
            x = torch.matmul(x.view(-1, 1), rep).cuda()
        else:
            x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)
