# IISC_Summer_Internship
This is an individual project which aims to predict polymer structures given the trajectory data of the polymer for a given time. Two research papers were implemented to achieve the required result.
 
The first research paper used a Variational Autoencoders Network to find an initial Reaction Coordinate and iteratively run biased MD simulations to find the optimal Reaction Coordinate.

The second research paper used an Autoencoder model with a Mixture Density Network to provide a reduced-order representation and then an LSTM-MDN model is used to forecast the latent space dynamics to maximise the latent data likelihood. The high dimensional state configurations are then obtained from the decoder.
