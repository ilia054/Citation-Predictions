import torch
import torch.nn as nn
import torch_optimizer as optim
import config as con

class Generator(nn.Module):
    """
    Generator neural network class for generating node embeddings.
    Inherits from nn.Module which is a base class for all neural network modules.
    """
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # Define the model layers using a Sequential container.
        self.model = nn.Sequential(
            nn.Linear(latent_dim, con.EMBEDDING_DIMENSION), # Linear layer with ReLU activation
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization        
            nn.Linear(con.EMBEDDING_DIMENSION, con.EMBEDDING_DIMENSION),   # Output layer
        )

    def forward(self, z):
        """
        Forward pass through the network.
        Args:
            z (Tensor): Input noise tensor to generate embeddings.
        Returns:
            Tensor: Generated embeddings.
        """
        return self.model(z)

class Discriminator(nn.Module):
    """
    Discriminator neural network class for discriminating between real and generated embeddings.
    Inherits from nn.Module.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
         # Define the model layers.
        self.model = nn.Sequential(
            nn.Linear(con.EMBEDDING_DIMENSION, con.EMBEDDING_DIMENSION // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),           
            nn.Linear(con.EMBEDDING_DIMENSION // 2, con.EMBEDDING_DIMENSION // 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(con.EMBEDDING_DIMENSION // 4, 1), 
            nn.Sigmoid()  # Output layer with sigmoid activation for binary classification
        )

    def forward(self, features):
        """
        Forward pass through the network.
        Args:
            features (Tensor): Node embeddings.
        Returns:
            Tensor: Probability of embeddings being real.
        """
        return self.model(features)

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss doesn't improve.
    """
    def __init__(self, patience=10, min_delta=0.01, save_embeddings=False):
        """
        Args:
            patience (int): Number of epochs to wait after min has been hit.
                            Training will stop if no improvement after this many epochs.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_embeddings (bool): Flag to determine whether to save embeddings.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_embeddings = save_embeddings
        self.best_embeddings = None

    def __call__(self, val_loss, current_embeddings=None):
        """
        Call method updates the state of early stopping upon each validation.
        Args:
            val_loss (float): Current epoch's validation loss.
            current_embeddings (Tensor, optional): Current embeddings for saving if improved.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_embeddings and current_embeddings is not None:
                self.best_embeddings = current_embeddings
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_embeddings and current_embeddings is not None:
                self.best_embeddings = current_embeddings
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stopping triggered')
                self.early_stop = True

def pre_train_G(graph, real_neighbor_pairs):
    """
    Pre-train the generator with given real neighbor pairs.
    Args:
        graph (Graph): Graph object containing nodes and edges.
        real_neighbor_pairs (Tensor): Tensor of real neighbor pairs for training.
    Returns:
        Tuple: Best embeddings if saved, generator model, optimizer, and loss function.
    """
    latent_dim = con.EMBEDDING_DIMENSION
    generator = Generator(latent_dim)
    # Initialize the RAdam optimizer for the generator
    optimizer_G = optim.RAdam(generator.parameters(), lr=con.G_LR)
    # Use MAE (L1 Loss) as the loss function
    # you can also use a different loss function to see different results
    # loss_function = nn.L1Loss()  
    loss_function = nn.MSELoss()    
    early_stopping = EarlyStopping(patience=int(graph.number_of_nodes() ** 0.5), min_delta=0.01, save_embeddings=True)

    print("Start Generator Pre-training Phase")
    for epoch in range(con.NUM_EPOCHS):  # Number of epochs
        # Generate noise vectors
        z = torch.randn(real_neighbor_pairs.size(0), latent_dim, device=real_neighbor_pairs.device)  # Ensure device compatibility

        # Generate fake embeddings from noise
        generated_embeddings = generator(z)

        # Calculate the MAE loss between generated embeddings and real neighbor pairs
        loss_G = loss_function(generated_embeddings, real_neighbor_pairs)

        # Early stopping check
        early_stopping(loss_G.item(), generated_embeddings)  # Pass loss.item() for early stopping
        if early_stopping.early_stop:
            print("Stopping training...")
            break

        # Backpropagation
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print("End Generator Pre-training Phase")

    return early_stopping.best_embeddings if early_stopping.save_embeddings else None, generator, optimizer_G, loss_function

def pre_train_D(real_neighbor_pairs, fake_embeddings):
    """
    Pre-train the discriminator to distinguish between real and fake (generated) embeddings.
    
    Args:
        real_neighbor_pairs (Tensor): Tensor containing embeddings of real neighbor pairs.
        fake_embeddings (Tensor): Tensor containing generated (fake) embeddings.
    
    Returns:
        Tuple: Returns the trained discriminator, its optimizer, and the loss function used.
    """
    # Initialize the discriminator model
    discriminator = Discriminator()
    # Use RAdam optimizer for the discriminator with learning rate from config
    optimizer_D = optim.RAdam(discriminator.parameters(), lr=con.LR)
    # Binary Cross Entropy Loss for binary classification task (real vs fake)
    loss_function = nn.BCELoss()
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)  # Adjust patience as needed

    # Create labels for real and fake data
    real_labels = torch.ones(real_neighbor_pairs.size(0), 1)
    fake_labels = torch.zeros(fake_embeddings.size(0), 1)
    
    # Combine real and fake embeddings and labels
    combined_embeddings = torch.cat([real_neighbor_pairs, fake_embeddings], dim=0)
    combined_labels = torch.cat([real_labels, fake_labels], dim=0)

    # Pre-training loop
    print("Start Discriminator Pre-training Phase")
    for epoch in range(con.NUM_EPOCHS):
        # Forward pass to get discriminator outputs for combined embeddings
        predictions = discriminator(combined_embeddings)

        # Calculate loss
        loss = loss_function(predictions, combined_labels)

        # Backward and optimize
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

        # Early stopping check
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Stopping training...")
            break

    print("End Discriminator Pre-training Phase")

    return discriminator, optimizer_D, loss_function