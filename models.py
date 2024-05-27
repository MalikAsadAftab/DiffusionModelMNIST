import torch
from torch import nn

class ResidualConvBlock(nn.Module):
    #Convolutional Residual Block 
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        
        # Define the first convolutional layer with batch normalization and GELU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Define the second convolutional layer with batch normalization and GELU activation
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.handle_channel = None
        if self.is_res and not self.same_channels:
            # If residual connection and channel mismatch, define a 1x1 convolution to match channels
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
    def forward(self, x):
        # Apply the convolutional layers
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                # If channel mismatch, apply the 1x1 convolution
                x_out = x_out + self.handle_channel(x)
            else:
                # Add the input to the output (residual connection)
                x_out = x_out + x
            return x_out / 1.414  # Normalize the output

class UNetDown(nn.Module):
    # UNet downward block (i.e., contracting block)
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [
            ResidualConvBlock(in_channels, out_channels, True),  # Use skip-connection
            ResidualConvBlock(out_channels, out_channels),       # Regular residual block
            nn.MaxPool2d(2)                                      # Downsampling using MaxPool
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    # UNet upward block (i.e., expanding block)
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # Double height & width using transposed convolution
            ResidualConvBlock(out_channels, out_channels, True),  # Use residual-connection
            ResidualConvBlock(out_channels, out_channels)         # Regular residual block
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        # Concatenate the input with the skip connection
        return self.model(torch.cat([x, x_skip], dim=1))

class EmbedFC(nn.Module):
    #Fully Connected Embedding Layer
    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),  # Fully connected layer
            nn.GELU(),                        # GELU activation
            nn.Linear(hidden_dim, hidden_dim) # Another fully connected layer
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Reshape the input and apply the fully connected layers
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]

class ContextUnet(nn.Module):
    #Context UNet model

    #Args:
    #    in_channels (int): Number of channels in the input image
    #    height (int): Height of the input image
    #    width (int): Width of the input image
    #    n_feat (int): Number of initial features i.e., hidden channels to which the input image will be transformed
    #    n_cfeat (int): Number of context features i.e., class categories
    #    n_downs (int): Number of down (and up) blocks of UNet. Default: 2

    def __init__(self, in_channels, height, width, n_feat, n_cfeat, n_downs=2):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_downs = n_downs

        # Define initial convolutional block
        self.init_conv = ResidualConvBlock(in_channels, n_feat, True)
        
        # Define downward UNet blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2**i * n_feat, 2**(i + 1) * n_feat))
        
        # Define center layers
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height // 2**len(self.down_blocks), width // 2**len(self.down_blocks))),
            nn.GELU()
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2**n_downs * n_feat, 2**n_downs * n_feat, 
                               (height // 2**len(self.down_blocks), width // 2**len(self.down_blocks))),
            nn.GroupNorm(8, 2**n_downs * n_feat),
            nn.GELU()
        )
        
        # Define upward UNet blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UNetUp(2**(i + 1) * n_feat, 2**(i - 1) * n_feat))

        # Define final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )

        # Define time & context embedding blocks
        self.timeembs = nn.ModuleList([EmbedFC(1, 2**i * n_feat) for i in range(n_downs, 0, -1)])
        self.contextembs = nn.ModuleList([EmbedFC(n_cfeat, 2**i * n_feat) for i in range(n_downs, 0, -1)])

    def forward(self, x, t, c):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0:
                downs.append(down_block(x))
            else:
                downs.append(down_block(downs[-1]))
        up = self.up0(self.to_vec(downs[-1]))
        for up_block, down, contextemb, timeemb in zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs):
            up = up_block(up * contextemb(c) + timeemb(t), down)
        return self.final_conv(torch.cat([up, x], axis=1))



#Explanation:
#ResidualConvBlock Class:

#__init__: Initializes the convolutional layers with batch normalization and GELU activation. If is_res is true and the input and output channels are different, a 1x1 convolution is used to match the channels.
#forward: Applies the convolutional layers and, if is_res is true, adds the input to the output (residual connection) and normalizes the result.
#UNetDown Class:

#__init__: Initializes a downward UNet block using two residual convolutional blocks and a max-pooling layer.
#forward: Applies the downward block to the input.
#UNetUp Class:

#__init__: Initializes an upward UNet block using a transposed convolution and two residual convolutional blocks.
#forward: Concatenates the input with the skip connection and applies the upward block.
#EmbedFC Class:

#__init__: Initializes a fully connected embedding layer with GELU activation.
#forward: Reshapes the input and applies the fully connected layers.
#ContextUnet Class:

#__init__: Initializes the Context UNet model, defining the initial convolutional block, downward blocks, center layers, upward blocks, final convolutional layer, and embedding blocks for time and context.
#forward: Applies the initial convolution, downward blocks, center layers, and upward blocks, combining the results with the skip connections and the embeddings for time and context. Finally, applies the final convolution to produce the output.