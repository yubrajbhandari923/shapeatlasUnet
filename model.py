from monai.networks.blocks import Convolution, MaxAvgPool, MLPBlock
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers import SkipConnection
from torch import nn

import warnings
from collections.abc import Sequence
import logging

# import
import torch
  
class UNetWithPrior(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] = 3,
        up_kernel_size: Sequence[int] = 3,
        num_res_units: int = 0,
        act: tuple  = Act.PRELU,
        norm: tuple  = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        prob_map: torch.Tensor = None,
        prob_map_latent_channels: int = 0,
        prob_map_encoder: nn.Module = None,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        
        self.prob_map = prob_map
        self.prob_map_latent_channels = prob_map_latent_channels
        
        self.prob_map_encoder = prob_map_encoder

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.latent_block = None
        
        self.down_blocks.append(self._get_down_layer(in_channels, channels[0], strides[0], True))
        self.up_blocks.insert(0, self._get_up_layer(channels[0] * 2, out_channels, strides[0], True))
        
        for i in range(1, len(channels) - 1):
            self.down_blocks.append(self._get_down_layer(channels[i - 1], channels[i], strides[i], False))
            self.up_blocks.insert(0, self._get_up_layer(channels[i] * 2, channels[i - 1], strides[i], False))
        
        # self.up_blocks.insert(0, self._get_up_layer(channels[-1] + 8, channels[-2], 1, False))         
        self.latent_block2 = self._get_up_layer(channels[-1] + self.prob_map_latent_channels, channels[-2], 1, False) 
        
        self.latent_block = self._get_bottom_layer(channels[-2], channels[-1])
        
        # self.latent_block2 = self._get_bottom_layer(channels[-1] + 8, channels[-1])
        
        # # print(f"Down Blocks: {self.down_blocks}")
        # print(f"Latent Block: {self.latent_block}")
        # print(f"Latent Block2: {self.latent_block2}")
        
        # print(f"Up Blocks: {self.up_blocks}")
        # print(f"Len Blocks: {len(self.up_blocks)}, {len(self.down_blocks)}")
        

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        # if self.num_res_units > 0:
        #     mod = ResidualUnit(
        #         self.dimensions,
        #         in_channels,
        #         out_channels,
        #         strides=strides,
        #         kernel_size=self.kernel_size,
        #         subunits=self.num_res_units,
        #         act=self.act,
        #         norm=self.norm,
        #         dropout=self.dropout,
        #         bias=self.bias,
        #         adn_ordering=self.adn_ordering,
        #     )
        #     return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        # if self.num_res_units > 0:
        #     ru = ResidualUnit(
        #         self.dimensions,
        #         out_channels,
        #         out_channels,
        #         strides=1,
        #         kernel_size=self.kernel_size,
        #         subunits=1,
        #         act=self.act,
        #         norm=self.norm,
        #         dropout=self.dropout,
        #         bias=self.bias,
        #         last_conv_only=is_top,
        #         adn_ordering=self.adn_ordering,
        #     )
        #     conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """
        if self.prob_map is not None:
            prob_map = self.prob_map_encoder(self.prob_map)
        
        skips = []
        
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            
        x = self.latent_block(x)
        
        if self.prob_map is not None:
            # logging.info(x.shape, prob_map.shape)
            prob_map = prob_map.repeat(x.shape[0], 1, 1, 1, 1)
            x = torch.cat((x, prob_map), dim=1)
        
        # print(x.shape)
        x = self.latent_block2(x)
        
        # print(x.shape, prob_map.shape)
        # x = torch.cat((x, prob_map), dim=1)

        for skip, up_block in zip(reversed(skips), self.up_blocks):            
            x = torch.cat((skip, x), dim=1)
            
            x = up_block(x)

        return x
  
class UNetWithLastLayerPrior(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] = 3,
        up_kernel_size: Sequence[int] = 3,
        num_res_units: int = 0,
        act: tuple  = Act.PRELU,
        norm: tuple  = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        prob_map: torch.Tensor = None,
        prob_map_latent_channels: int = 0,
        prob_map_encoder: nn.Module = None,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        
        self.prob_map = prob_map
        self.prob_map_latent_channels = prob_map_latent_channels
        
        self.prob_map_encoder = prob_map_encoder

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.latent_block = None
        
        self.down_blocks.append(self._get_down_layer(in_channels, channels[0], strides[0], True))
        self.up_blocks.insert(0, self._get_up_layer(channels[0] * 2 + self.prob_map_latent_channels, out_channels, strides[0], True))
        
        
        
        for i in range(1, len(channels) - 1):
            self.down_blocks.append(self._get_down_layer(channels[i - 1], channels[i], strides[i], False))
            self.up_blocks.insert(0, self._get_up_layer(channels[i] * 2, channels[i - 1], strides[i], False))
        
        # self.up_blocks.insert(0, self._get_up_layer(channels[-1] + channels[-2], channels[-3], 1, False))         
        self.latent_block2 = self._get_up_layer(channels[-1], channels[-2], 1, False) 
        
        self.latent_block = self._get_bottom_layer(channels[-2], channels[-1])
        
        # self.latent_block2 = self._get_bottom_layer(channels[-1] + 8, channels[-1])
        
        # # print(f"Down Blocks: {self.down_blocks}")
        # print(f"Latent Block: {self.latent_block}")
        # print(f"Latent Block2: {self.latent_block2}")
        
        # print(f"Up Blocks: {self.up_blocks}")
        # print(f"Len Blocks: {len(self.up_blocks)}, {len(self.down_blocks)}")
        

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        # if self.num_res_units > 0:
        #     mod = ResidualUnit(
        #         self.dimensions,
        #         in_channels,
        #         out_channels,
        #         strides=strides,
        #         kernel_size=self.kernel_size,
        #         subunits=self.num_res_units,
        #         act=self.act,
        #         norm=self.norm,
        #         dropout=self.dropout,
        #         bias=self.bias,
        #         adn_ordering=self.adn_ordering,
        #     )
        #     return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        # if self.num_res_units > 0:
        #     ru = ResidualUnit(
        #         self.dimensions,
        #         out_channels,
        #         out_channels,
        #         strides=1,
        #         kernel_size=self.kernel_size,
        #         subunits=1,
        #         act=self.act,
        #         norm=self.norm,
        #         dropout=self.dropout,
        #         bias=self.bias,
        #         last_conv_only=is_top,
        #         adn_ordering=self.adn_ordering,
        #     )
        #     conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """
        if self.prob_map is not None:
            prob_map = self.prob_map_encoder(self.prob_map)
            prob_map = prob_map.repeat(x.shape[0], 1, 1, 1, 1)
        
        skips = []
        
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            
        x = self.latent_block(x)
        x = self.latent_block2(x)
        
        # if self.prob_map is not None:
        #     # logging.info(x.shape, prob_map.shape)
        #     x = torch.cat((x, prob_map), dim=1)
        
        # print(x.shape)
        
        # print(x.shape, prob_map.shape)
        # x = torch.cat((x, prob_map), dim=1)
        
        

        for skip, up_block in zip(reversed(skips[1:]), self.up_blocks[:-1]):            
            x = torch.cat((x, skip), dim=1)
            x = up_block(x)
        
        # print(f"Shapes: {prob_map.shape}, {x.shape}, {skips[0].shape}")
        
        x = torch.cat((prob_map, x, skips[0]), dim=1)
        x = self.up_blocks[-1](x)

        return x

#
# - Training Time (Smaller Dataset)
# - Smaller Dataset: 50, 100, 200, 500, 1000
# - Model Size
# - Model Accuracy