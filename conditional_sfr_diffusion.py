import numpy as np
import os
import torch
from score_models import ScoreModel, NCSNpp, VESDE
import getopt
import sys
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_device(DEVICE)
dtype = torch.float32
torch.set_default_dtype(dtype)

checkpoints_folder = '/checkpoints'
plot_folder = '/plots'

def get_sigma_max_min(save_path, data=None):
    """
    This function calculates the maximum and minimum sigma values from the given dataset.
    The maximum sigma value is the maximum distance between any two points in the dataset,
    and the minimum sigma value is the smallest non-zero value in the dataset (or 1e-30 if there is an issue).
    If the maximum and minimum sigma values are already saved in files, they are read from those files.
    Otherwise, they are calculated and saved to the respective files.

    Args:
        save_path (str): The directory path where the sigma values are saved or will be saved.
        data (torch.Tensor, optional): The dataset from which to calculate the sigma values. 
                                        Required if the sigma values are not already saved in files.
    Returns:
        tuple: A tuple containing two float values:
            - sigma_max (float): The maximum sigma value.
            - sigma_min (float): The minimum sigma value.
    """
    file_max = save_path + '/max_dist.txt'
    file_min = save_path + '/min_dist.txt'

    sigma_max = 0
    sigma_min = 0

    if os.path.isfile(file_max):
        with open(file_max, 'r') as f:
            sigma_max = float(f.read())
    else:
        max_dist = 0

        # This loop is inefficient but it should only be run once
        for i in range(len(data)):
            for j in range(i, len(data)):
                dist = torch.norm(data[i] - data[j])
                if dist > max_dist:
                    max_dist = dist

        with open(file_max, 'w') as f:
            r = max_dist
            if torch.is_tensor(r):
                r = r.item()
            f.write(str(r))
        
        sigma_max = max_dist.item()

    if os.path.exists(file_min):
        with open(file_min, 'r') as f:
            sigma_min = float(f.read())
    else:
        sigma_min = data[data>0].min().item()
        if sigma_min == 0 and dtype == torch.float32:
            # a small number within the range of float32
            sigma_min = 1e-30

        with open(file_min, 'w') as f:
            f.write(str(sigma_min))

    return sigma_max, sigma_min


def train_conditional_SSP(data, directory, sigma_min, sigma_max, time_only=False, ch_mult=[2, 2,], epochs=10000):
    """
    Trains a conditional Score-Based Model on the given dataset.

    Args:
        data (torch.utils.data.Dataset): The dataset to train the model on. It should be an instance of DatasetCond or DatasetCondTime.
        directory (str): The directory where the model checkpoints will be saved.
        sigma_min (float): The minimum sigma value for the SDE.
        sigma_max (float): The maximum sigma value for the SDE.
        time_only (bool, optional): If True, the model is conditioned only on time. Otherwise, it is conditioned on both time and metallicity. Default is False.
        ch_mult (list, optional): A list of integers representing the channel multiplier for each layer of the neural network. Default is [2, 2].
        epochs (int, optional): The number of epochs to train the model. Default is 10000.
    Returns:
        ScoreModel: The trained Score-Based Generative Model.
    """

    # add conditional channels to the net
    if time_only:
        conditions = (["time_continuous"])
    else:
        conditions = (["time_continuous", "time_continuous"])

    # These hyperparameters can be played around with
    net = NCSNpp(channels=1, dimensions=1, nf=16, ch_mult=ch_mult, conditions=conditions)
    sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
    model = ScoreModel(net, sde, device=DEVICE, checkpoints_directory=directory)

    model.fit(data, batch_size=256, learning_rate=1e-4, epochs=epochs)

    return model

class DatasetCond(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for conditional data.
    It encodes the conditional information with the SSP data.

    Attributes:
        data (numpy.ndarray or torch.Tensor): The main dataset.
        time (numpy.ndarray or torch.Tensor): The time condition associated with the data.
        metallicity (numpy.ndarray or torch.Tensor): The metallicity condition associated with the data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the data, time, and metallicity for a given index.
    """
    def __init__(self, data, time, metallicity):
        self.data = data
        self.time = time # conditional
        self.metallicity = metallicity # conditional

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.time[idx], self.metallicity[idx]

class DatasetCondTime(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for conditional data with time as the only condition.
    """
    def __init__(self, data, time):
        self.data = data
        self.time = time

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.time[idx]


def load_data_SSP_table(data_directory):
    """
    This function load in the SSP data from a directory and processes it
    TODO - this function must be modified to handle the desired data
    The SSPs should be a 3D tensor with dimensions (time, metallicity, D) where D is the spectra dimension
    NOTE - If the data already exists in a flattened format (time, metallicity, wavelength), then this function is not needed,
    But the data should be partitioned into train and test sets

    Args:
        data_directory (str): The directory where the SSP data is stored.
    Returns:
        tuple: A tuple containing three tensors:
            - table_tmw (torch.Tensor): The SSP data tensor with dimensions (time, metallicity, D) where D is the spectra dimension.
            - time_table (torch.Tensor): The tensor containing the time values.
            - metal_table (torch.Tensor): The tensor containing the metallicity values.
    """

    if not os.path.exists(data_directory):
        print("Data directory does not exist, returning")
        return

    # TODO - load the data
    # the data should be a big table indexed by time and metallicity, with the spectra as the values
    # and the time and metallicity used to generate the table should be returned as well
    # in the next function, these will be flattened out to be a list of (time, metallicity, wavelength), rather than a cube

    # tmw = time metallicity wavelength
    # placeholder
    table_tmw, time_table, metal_table = [], [], []

    time_table = torch.tensor(time_table, dtype=dtype)
    metal_table = torch.tensor(metal_table, dtype=dtype)
    table_tmw = torch.tensor(table_tmw, dtype=dtype)

    return table_tmw, time_table, metal_table

# This function generates a fake dataset for testing purposes

def generate_fake_data(data_size=64):
    """
    Generates fake linear SSP data for testing purposes. The data is generated as a linear combination of time and metallicity.
 
    Args:
        data_size (int, optional): The size of the data to generate. Defaults to 64.
    Returns:
        tuple: A tuple containing:
            - SSPs (torch.Tensor): A tensor of shape (n_data_points, n_data_points, data_size) 
              containing the generated SSP data.
            - t (torch.Tensor): A tensor of shape (n_data_points,) representing the time component.
            - m (torch.Tensor): A tensor of shape (n_data_points,) representing the metalliicity component.
    """

    mult_func = torch.arange(1, data_size+1)/10

    n_data_points = 100
    t = torch.arange(0, n_data_points)/(n_data_points/2)
    m = torch.arange(0, n_data_points)/(n_data_points/2)
    SSPs = torch.zeros(n_data_points, n_data_points, data_size)

    for i in range(n_data_points):
        for j in range(n_data_points):
            SSPs[i, j] = t[i]*mult_func + m[j]

    return SSPs, t, m

def generate_fake_data_time_only(data_size=4, n_data_points=100):
    """
    Generates fake linear SSP data for testing purposes. The data is dependent only on time.
    Metallicity is also generated for consistency but it is not used in training

    Args:
        data_size (int, optional): The size of the data to generate. Defaults to 64.
    Returns:
        tuple: A tuple containing:
            - SSPs (torch.Tensor): A tensor of shape (n_data_points, n_data_points, data_size) 
              containing the generated SSP data.
            - t (torch.Tensor): A tensor of shape (n_data_points,) representing the time component.
            - m (torch.Tensor): A tensor of shape (n_data_points,) representing the metallicity component.
    """

    mult_func = torch.arange(1, data_size+1)/10

    t = torch.arange(0, n_data_points)/(n_data_points/2)
    m = torch.arange(0, n_data_points)/(n_data_points/2)
    spectra = torch.zeros(n_data_points, data_size)

    for i in range(n_data_points):
        spectra[i] = t[i]*mult_func

    return spectra, t, m

def partition_data_set(data_directory, train_ratio=0.8, padding=0, linear_test=False, time_only=False):
    """
    Partitions a dataset into training and testing sets.

    Args:
        data_directory (str): The directory where the data is stored.
        train_ratio (float, optional): The ratio of the dataset to be used for training. Default is 0.8.
        padding (int, optional): The amount of padding to add to the data. For use in the case the SSPs have an awkward shape
                                 (ie - fsps has 5994 dimensions so adding 6 makes it 6000, which is easier to work with).
                                 Default is 0.
        linear_test (bool, optional): If True, generates a fake dataset for linear testing. Default is False.
        time_only (bool, optional): If True, only uses time data for the dataset. Default is False.

    Returns:
        tuple: A tuple containing:
            - train_data (torch.Tensor): The training SSP data.
            - time_train (torch.Tensor): The training time data.
            - metal_train (torch.Tensor): The training metallicity data.
            - test_data (torch.Tensor): The testing SSP data.
            - time_test (torch.Tensor): The testing time data.
            - metal_test (torch.Tensor): The testing metallicity data.
    """

    if linear_test:
        if time_only:
            ssp_table, time_table, metal_table = generate_fake_data_time_only()
        else:
            ssp_table, time_table, metal_table = generate_fake_data()
    else:
        ssp_table, time_table, metal_table = load_data_SSP_table(data_directory)

    time_table = torch.tensor(time_table, dtype=dtype)
    metal_table = torch.tensor(metal_table, dtype=dtype)
    ssp_table = torch.tensor(ssp_table, dtype=dtype)

    if time_only:
        full_SSPs = torch.zeros((time_table.shape[0], 1, ssp_table.shape[-1] + padding), device=DEVICE)
    else:
        full_SSPs = torch.zeros((time_table.shape[0]*metal_table.shape[0], 1, ssp_table.shape[-1] + padding), device=DEVICE)
    time = []
    metal = []

    # These loops are quite ugly but it should only be run once to generate the dataset
    if time_only:
        full_SSPs = ssp_table.clone().reshape(full_SSPs.shape)
        time_tensor = time_table.clone()
        metal_tensor = metal_table.clone()
    else:
        for t in range(time_table.shape[0]):
            for z in range(metal_table.shape[0]):
                if padding > 0:
                    ssp = torch.cat((ssp_table[t, z], torch.zeros(padding)))
                else:
                    ssp = ssp_table[t, z]

                full_SSPs[t*metal_table.shape[0] + z][0] = ssp
                time.append(time_table[t])
                metal.append(metal_table[z])

        time_tensor = torch.tensor(time, device=DEVICE)
        metal_tensor = torch.tensor(metal, device=DEVICE)

    # Define split sizes
    train_size = int(train_ratio * full_SSPs.size(0))
    # Randomly shuffle indices
    indices = torch.randperm(full_SSPs.size(0))

    # Split the indices for training and testing
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create the train and test splits
    train_data = full_SSPs[train_indices]
    test_data = full_SSPs[test_indices]

    train_indices = train_indices.to(DEVICE)
    test_indices = test_indices.to(DEVICE)

    # Perform safe indexing
    time_train = time_tensor[train_indices]
    metal_train = metal_tensor[train_indices]

    time_test = time_tensor[test_indices]
    metal_test = metal_tensor[test_indices]

    return train_data, time_train, metal_train, test_data, time_test, metal_test

def train_conditional(save_directory, data_directory=None, normalize=False, linear_test=False, time_only=False, log_scale=True, epochs=10000, sample=True):
    """
    Trains a conditional model to generate SSPs based on time/metallicity and saves the trained model and generated samples.

    Args:
        save_directory (str): Directory to save the model, plots, and datasets.
        data_directory (str, optional): Directory containing the data. Defaults to None.
        normalize (bool, optional): Whether to normalize the training data. Defaults to False.
        linear_test (bool, optional): Whether to use linear test data. Defaults to False.
        time_only (bool, optional): Whether to use only time as a condition. Defaults to False.
        log_scale (bool, optional): Whether to use logarithmic scale for plotting. Defaults to True.
        epochs (int, optional): Number of training epochs. Defaults to 10000.
        sample (bool, optional): Whether to generate and save samples after training. Defaults to True.

    Returns:
        model: The trained model.
    """

    train_file_name  = save_directory + '/train_dataset.pt'
    test_file_name = save_directory + '/test_dataset.pt'
    time_train_file_name = save_directory + '/time_train.pt'
    metal_train_file_name = save_directory + '/metal_train.pt'
    time_test_file_name = save_directory + '/time_test.pt'
    metal_test_file_name = save_directory + '/metal_test.pt'

    plot_dir = save_directory + plot_folder
    checkpoints_directory = save_directory + checkpoints_folder

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    if not os.path.exists(train_file_name):
        train_data, time_train, metal_train, test_data, time_test, metal_test = partition_data_set(data_directory, linear_test=linear_test, time_only=time_only)

        torch.save(train_data, train_file_name)
        torch.save(test_data, test_file_name)
        torch.save(time_train, time_train_file_name)
        torch.save(metal_train, metal_train_file_name)
        torch.save(time_test, time_test_file_name)
        torch.save(metal_test, metal_test_file_name)

    else:
        train_data = torch.load(train_file_name, map_location=DEVICE, weights_only=True)
        test_data = torch.load(test_file_name, map_location=DEVICE, weights_only=True)
        time_train = torch.load(time_train_file_name, map_location=DEVICE, weights_only=True)
        metal_train = torch.load(metal_train_file_name, map_location=DEVICE, weights_only=True)
        time_test = torch.load(time_test_file_name, map_location=DEVICE, weights_only=True)
        metal_test = torch.load(metal_test_file_name, map_location=DEVICE, weights_only=True)

    if linear_test:
        wave = torch.arange(0, train_data.shape[-1]).cpu().numpy()
    elif os.path.exists(save_directory + "/wavelengths.pt"):
        wave = torch.load(save_directory + "/wavelengths.pt", map_location=DEVICE, weights_only=True)
    else:
        raise FileNotFoundError("No wavelength file found")

    # normalize the train_data
    if normalize:
        mean = train_data.mean()
        std = train_data.std()
        train = (train_data - mean) / std
    else:
        train = train_data

    if time_only:
        ssp_dataset = DatasetCondTime(train, time_train)
    else:
        ssp_dataset = DatasetCond(train, time_train, metal_train)

    # These values can be played around with
    sigma_max, sigma_min = get_sigma_max_min(save_directory, train)

    # the data must be divisable by ch_mult
    if train_data.shape[-1] == 4:
        ch_mult = [2, 2]
    else:
        ch_mult = [2, 2, 2, 2]

    model = train_conditional_SSP(ssp_dataset, checkpoints_directory, sigma_min, sigma_max, time_only=time_only, ch_mult=ch_mult, epochs=epochs)

    # save the model
    torch.save(model.state_dict(), checkpoints_directory + "/conditional_model.pt")

    if not sample:
        return model

    # get a random train point
    train_idx = np.random.randint(0, len(train_data))
    ssp_train = train_data[train_idx]
    t_train = time_train[train_idx]
    z_train = metal_train[train_idx]
    num_samples = 5

    pt = t_train.repeat(5, 1)
    pz = z_train.repeat(5, 1)
    if time_only:
        train_samples = model.sample(pt, shape=[5, 1, train.shape[-1]], steps=5000).reshape(num_samples, train.shape[-1])
    else:
        train_samples = model.sample(pt, pz, shape=[5, 1, train.shape[-1]], steps=5000).reshape(num_samples, train.shape[-1])

    min_idx = 0
    plotted_files = os.listdir(plot_dir)
    while plotted_files.count('conditional_' + str(train_idx) + "_" + str(min_idx) + '_train_sample.png') > 0:
        min_idx += 1

    # plot all train samples vs wave
    ssp_train = ssp_train.squeeze().cpu().numpy()
    if not linear_test and log_scale:
        plt.xscale('log')

    plt.plot(wave, ssp_train, color='green', label='True train')

    for samp in train_samples:
        plt.plot(wave, samp.cpu().numpy())

    print("The conditional train index is", train_idx)
    plt.plot(wave, ssp_train, color='green')
    plt.legend()
    plt.savefig(plot_dir + '/conditional_' + str(train_idx) + "_" + str(min_idx) + '_train_sample.png')
    plt.clf()

    # get a random test point
    test_idx = np.random.randint(0, len(test_data))
    ssp_test = test_data[test_idx]
    t_test = time_test[test_idx]
    z_test = metal_test[test_idx]

    pt = t_test.repeat(5, 1)
    pz = z_test.repeat(5, 1)
    if time_only:
        test_samples = model.sample(pt, shape=[5, 1, test_data.shape[-1]], steps=5000).reshape(num_samples, test_data.shape[-1])
    else:
        test_samples = model.sample(pt, pz, shape=[5, 1, test_data.shape[-1]], steps=5000).reshape(num_samples, test_data.shape[-1])

    min_idx = 0
    plotted_files = os.listdir(plot_dir)
    while plotted_files.count('conditional_' + str(test_idx) + "_" + str(min_idx) + '_test_sample.png') > 0:
        min_idx += 1

    # plot all train samples vs wave
    ssp_test = ssp_test.squeeze().cpu().numpy()
    if not linear_test and log_scale:
        plt.xscale('log')

    plt.plot(wave, ssp_test, color='green', label='True test')

    for samp in test_samples:
        plt.plot(wave, samp.cpu().numpy())

    print("The conditional test index is", test_idx)
    plt.plot(wave, ssp_test, color='green')
    plt.legend()
    plt.savefig(plot_dir + '/conditional_' + str(test_idx) + "_" + str(min_idx) + '_test_sample.png')
    plt.clf()

    return model


colours = ["deepskyblue", "firebrick", "orange", "pink", "blueviolet"]
def sample_conditional_model(save_directory, normalize=False, linear_test=False, time_only=False, log_scale=True):
    """
    Samples a conditional model and plots the results.
    Args:
        save_directory (str): The directory where the model checkpoints and data are saved.
        normalize (bool, optional): If True, the samples will be un-normalized to compare to the original data. Default is False.
        linear_test (bool, optional): If True, the wavelengths for the linear data test are used. Default is False.
        time_only (bool, optional): If True, the model is only conditioned on time. Default is False.
        log_scale (bool, optional): If True, the x-axis of the plots will be in log scale. Default is True.
    Raises:
        FileNotFoundError: If the wavelength file is not found in the save_directory.
    Returns:
        None
    """

    checkpoints_directory = save_directory + checkpoints_folder
    plot_dir = save_directory + plot_folder

    train_data = torch.load(save_directory + '/train_dataset.pt', map_location=DEVICE, weights_only=True)
    test_data = torch.load(save_directory + '/test_dataset.pt', map_location=DEVICE, weights_only=True)
    time_train = torch.load(save_directory + '/time_train.pt', map_location=DEVICE, weights_only=True)
    metal_train = torch.load(save_directory + '/metal_train.pt', map_location=DEVICE, weights_only=True)
    time_test = torch.load(save_directory + '/time_test.pt', map_location=DEVICE, weights_only=True)
    metal_test = torch.load(save_directory + '/metal_test.pt', map_location=DEVICE, weights_only=True)

    if linear_test:
        wave = torch.arange(0, train_data.shape[-1])
        wave = wave.cpu().numpy()
    elif os.path.exists(save_directory + "/wavelengths.pt"):
        wave = torch.load(save_directory + "/wavelengths.pt", weights_only=True, map_location=DEVICE)
        wave = wave.cpu().numpy()
    else:
        raise FileNotFoundError("No wavelength file found")

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plotted_files = os.listdir(plot_dir)

    # get a random train point
    train_idx = np.random.randint(0, len(train_data))
    ssp_train = train_data[train_idx]
    t_train = time_train[train_idx]
    z_train = metal_train[train_idx]

    # get a random test point
    test_idx = np.random.randint(0, len(test_data))
    ssp_test = test_data[test_idx]
    t_test = time_test[test_idx]
    z_test = metal_test[test_idx]

    model = ScoreModel(checkpoints_directory=checkpoints_directory)

    num_samples = 5

    if time_only:
        train_samples = model.sample(t_train.repeat(num_samples, 1), shape=[num_samples, 1, ssp_test.shape[-1]], steps=1000).reshape(num_samples, ssp_test.shape[-1])
        test_samples = model.sample(t_test.repeat(num_samples, 1), shape=[num_samples, 1, ssp_test.shape[-1]], steps=1000).reshape(num_samples, ssp_test.shape[-1])
    else:
        train_samples = model.sample(t_train.repeat(num_samples, 1), z_train.repeat(num_samples, 1), shape=[num_samples, 1, ssp_test.shape[-1]], steps=5000).reshape(num_samples, ssp_test.shape[-1])
        test_samples = model.sample(t_test.repeat(num_samples, 1), z_test.repeat(num_samples, 1), shape=[num_samples, 1, ssp_test.shape[-1]], steps=5000).reshape(num_samples, ssp_test.shape[-1])

    # If model is normalized then un-normalize the samples to compare to original
    if normalize:
        mean = train_data.mean()
        std = train_data.std()
        train_samples = train_samples*std + mean
        test_samples = test_samples*std + mean

    min_idx = 0
    while plotted_files.count('conditional_' + str(train_idx) + "_" + str(min_idx) + '_train_sample.png') > 0:
        min_idx += 1

    # plot all train samples vs wave
    ssp_train = ssp_train.cpu().numpy().squeeze()
    ssp_test = ssp_test.cpu().numpy().squeeze()

    if not linear_test and log_scale:
        plt.xscale('log')

    plt.plot(wave, ssp_train, color='green', label='True train')

    for i in range(len(train_samples)):
        samp = train_samples[i].cpu().numpy()
        if len(train_samples) == 5:
            plt.plot(wave, samp, color=colours[i])
        else:
            # let matplotlib decide the colours
            plt.plot(wave, samp)

    print("The conditional train index is", train_idx)
    plt.plot(wave, ssp_train, color='green')
    plt.savefig(plot_dir + '/conditional_' + str(train_idx) + "_" + str(min_idx) + '_train_sample.png')
    plt.legend()
    plt.clf()

    min_idx = 0
    while plotted_files.count('conditional_' + str(test_idx) + "_" + str(min_idx) + '_test_sample.png') > 0:
        min_idx += 1

    if not linear_test and log_scale:
        plt.xscale('log')
    plt.plot(wave, ssp_test, color='green', label='True test')
    for i in range(len(test_samples)):
        samp = test_samples[i].cpu().numpy()
        if len(test_samples) == 5:
            plt.plot(wave, samp, color=colours[i])
        else:
            plt.plot(wave, samp)

    print("The conditional test index is", test_idx)
    plt.plot(wave, ssp_test, color='green')
    plt.savefig(plot_dir + '/conditional_'+ str(test_idx) + "_" + str(min_idx) + '_test_sample.png')
    plt.legend()
    plt.clf()


def main():
    argv = sys.argv[1:]
    # Where the model checkpoints should be saved
    save_directory = None
    # Where the data is stored
    data_directory = None
    # Whether or not to generate samples from the model
    generate_samples = False
    # Whether or not to normalize the data
    normalize = False
    # The model is conditioned only on time
    time_only = False

    try:
        opts, args = getopt.getopt(argv, "gsdnt", ["generate=", "save=", "data=", "normalize=", "time_only="])

        for opt, arg in opts:
            # Where the data and model checkpoints should be saved
            if opt in ['-s', '--save']:
                save_directory = arg

            # If this option is provided, the model will be sampled instead of trained. The model must already exist in this case
            elif opt in ['-g', '--generate']:
                generate_samples = True

            elif opt in ['-d', '--data']:
                data_directory = arg

            # Whether the data should be normalized before training
            elif opt in ['-n', '--normalize']:
                normalize = True

            # The model is conditioned only on time
            elif opt in ['-t', '--time_only']:
                time_only = True

    except:
        print("Error in arguments, returning")
        return

    if save_directory is None:
        print("No save directory specified, returning")
        return

    if generate_samples:
        sample_conditional_model(save_directory, normalize=normalize, linear_test=False, time_only=time_only)
    else:
        train_conditional(save_directory, data_directory, normalize=normalize, linear_test=False, time_only=time_only)

if __name__ == "__main__":
    main()

