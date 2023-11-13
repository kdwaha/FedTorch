from src import *
from src.clients import Aggregator
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from src.train.train_utils import *
from src.utils.hessian import hessian

import random
import ray
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def model_download(aggregator: Aggregator, clients: dict) -> None:
    """
    Client download the model from the aggregator.
    Args:
        aggregator: (class Aggregator) Model aggregator for federated learning.
        clients: (dict) client list of Client class.
    Returns: None
    """
    model_weights = aggregator.get_parameters()
    for k, client in clients.items():
        client.model = model_weights


def model_collection(clients: dict, aggregator: Aggregator, with_data_len: bool = False) -> None:
    """
    Clients uploads each model to aggregator.
    Args:
        clients: (dict) {str: Client class} form.
        aggregator: (Aggregator) Aggregator class.
        with_data_len: (bool) Data length return if True.

    Returns: None

    """
    collected_weights = {}
    for k, client in clients.items():
        if with_data_len:
            collected_weights[k] = {'weights': client.get_parameters(),
                                    'data_len': client.data_len()}
        else:
            collected_weights[k] = {'weights': client.get_parameters()}
    aggregator.collected_weights = collected_weights


def compute_accuracy(model: Module,
                     data_loader: DataLoader,
                     loss_fn: Optional[Module] = None,
                     **kwargs) -> Union[float, tuple]:
    """
    Compute the accuracy using its whole data.

    Args:
        model: (torch.Module) Training model.
        data_loader: (torch.utils.Dataloader) Dataloader.
        loss_fn: (torch.Module) Optional. Loss function.

    Returns: ((float) accuracy, (float) loss)

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    global_model = None
    if 'global_model' in kwargs:
        global_model = kwargs['global_model']
        global_model.to(device)
        global_model.eval()

    correct = []
    loss_list = []
    total_len = []

    i = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).to(torch.long)

            if hasattr(model, 'output_feature_map') and model.output_feature_map:
                outputs, feature_map = model(x)
            else:
                outputs = model(x)

            if global_model is not None:
                if hasattr(model, 'output_feature_map'):
                    g_outputs, _ = global_model(x)
                else:
                    g_outputs = global_model(x)

            if loss_fn is not None:
                if 'FeatureBalanceLoss' in str(loss_fn.__class__):
                    loss = loss_fn(outputs, g_outputs, y, feature_map, i)
                    loss_list.append(loss.item())
                else:
                    loss = loss_fn(outputs, y)
                    loss_list.append(loss.item())

            y_max_scores, y_max_idx = outputs.max(dim=1)
            correct.append((y == y_max_idx).sum().item())
            total_len.append(len(x))
            i += 1
        acc = sum(correct) / sum(total_len)

        if loss_fn is not None:
            loss = sum(loss_list) / sum(total_len)
            return acc, loss
        else:
            return acc


def client_sampling(clients: dict, sample_ratio: float, global_round: int) -> list:
    """
    Sampling the client from total clients.
    Args:
        clients: (dict) Clients dictionary that have all the client instances.
        sample_ratio: (float) Sample ration.
        global_round: (int) Current global round.

    Returns: (list) Sample 'client class'

    """
    sampled_clients = random.sample(list(clients.values()), k=int(len(clients.keys()) * sample_ratio))
    for client in sampled_clients:
        # NOTE: I think it is purpose to check what clients are joined corresponding global iteration.
        client.global_iter.append(global_round)
    return sampled_clients


def update_client_dict(clients: dict, trained_client: list) -> dict:
    """
    Updates the client dictionary. Only trained client are updated.
    Args:
        clients: (dict) Clients dictionary.
        trained_client: (list) Trained clients.

    Returns: (dict) Clients dictionary.

    """
    for client in trained_client:
        clients[client.name] = client
    return clients


def mark_accuracy(model_l: Module, model_g: Module, dataloader: DataLoader,
                  summary_writer: SummaryWriter, tag: str, epoch: int) -> None:
    """
    Accuracy mark for experiment.
    Args:
        model_l: (torch.Module) Local model.
        model_g: (torch.Module) Global model.
        dataloader: (DataLoader) Client's dataloader.
        summary_writer: (SummaryWriter class) SummaryWriter instance.
        tag: (str) Summary tag.
        epoch: (str) Client epochs.

    Returns: (None)

    """
    device = "cpu"

    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    accuracy_l = compute_accuracy(model_l, dataloader)
    accuracy_g = compute_accuracy(model_g, dataloader)

    summary_writer.add_scalar('{}/accuracy/local_model'.format(tag), accuracy_l, epoch)
    summary_writer.add_scalar('{}/accuracy/global_model'.format(tag), accuracy_g, epoch)


def mark_entropy(model_l: Module, model_g: Module, dataloader: DataLoader,
                 summary_writer: SummaryWriter, epoch: int) -> None:
    """
    Mark the entropy and entropy gap from certain dataloader.
    Args:
        model_l: (torch.Module) Local Model
        model_g: (torch.Module) Global Model
        dataloader: (DataLoader) Dataloader either train or test.
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.

    Returns: (None)

    """
    device = "cpu"
    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    output_entropy_l_list = []
    output_entropy_g_list = []
    feature_entropy_l_list = []
    feature_entropy_g_list = []

    output_gap_list = []
    feature_gap_list = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs_l = model_l(x)
            outputs_g = model_g(x)

            features_l = model_l.feature_maps(x)
            features_g = model_g.feature_maps(x)

            # INFO: Calculates entropy gap
            # INFO: Output
            # Probability distributions
            output_prob_l = get_probability(outputs_l, logit=True)
            output_prob_g = get_probability(outputs_g, logit=True)

            # INFO: Output entropy
            output_entr_l = entropy(output_prob_l, base='exp')
            output_entr_g = entropy(output_prob_g, base='exp')

            # Insert the original output entropy value
            output_entropy_l_list.append(torch.mean(output_entr_l))
            output_entropy_g_list.append(torch.mean(output_entr_g))

            # INFO: Calculates the entropy gap
            outputs_entr_gap = torch.abs(output_entr_g - output_entr_l)
            output_gap_list.append(torch.mean(outputs_entr_gap))

            # INFO: Feature
            # Probability distributions
            feature_prob_l = get_probability(features_l)
            feature_prob_g = get_probability(features_g)

            # Calculate the entropy
            feature_entr_l = entropy(feature_prob_l)
            feature_entr_g = entropy(feature_prob_g)

            # Insert the value
            feature_entropy_l_list.append(sum_mean(feature_entr_l))
            feature_entropy_g_list.append(sum_mean(feature_entr_g))

            # INFO: Calculates the entropy gap
            feature_entr_gap = torch.abs(feature_entr_g - feature_entr_l)
            feature_gap_list.append(sum_mean(feature_entr_gap))

    output_entropy_l = torch.Tensor(output_entropy_l_list)
    output_entropy_g = torch.Tensor(output_entropy_g_list)
    feature_entropy_l = torch.Tensor(feature_entropy_l_list)
    feature_entropy_g = torch.Tensor(feature_entropy_g_list)

    summary_writer.add_scalar("entropy/feature/local", torch.mean(feature_entropy_l), epoch)
    summary_writer.add_scalar("entropy/feature/global", torch.mean(feature_entropy_g), epoch)
    summary_writer.add_scalar("entropy/classifier/local", torch.mean(output_entropy_l), epoch)
    summary_writer.add_scalar("entropy/classifier/global", torch.mean(output_entropy_g), epoch)

    output_gap = torch.Tensor(output_gap_list)
    feature_gap = torch.Tensor(feature_gap_list)
    summary_writer.add_scalar("entropy_gap/classifier", torch.mean(output_gap), epoch)
    summary_writer.add_scalar("entropy_gap/feature", torch.mean(feature_gap), epoch)


def mark_norm_gap(model_l: Module, model_g: Module, dataloader: DataLoader,
                  summary_writer: SummaryWriter, epoch: int, norm: int = 1, prob: bool = False) -> None:
    """
    Mark the norm gap from certain dataloader.
    Args:
        model_l: (torch.Module) Local Model
        model_g: (torch.Module) Global Model
        dataloader: (DataLoader) Dataloader either train or test.
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.
        norm: (int) Level of norm value. Default is 1.
        prob: (bool) To check to use probability form when calculate the norm.

    Returns: (None)

    """
    device = "cpu"
    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    outputs_norm_gap_list = []
    features_norm_gap_list = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs_l = model_l(x)
            outputs_g = model_g(x)

            features_l = model_l.feature_maps(x)
            features_g = model_g.feature_maps(x)

            # INFO: Calculates entropy gap

            # INFO: Output
            # Probability distributions
            if prob:
                outputs_l = get_probability(outputs_l, logit=True)
                outputs_g = get_probability(outputs_g, logit=True)

                features_l = get_probability(features_l)
                features_g = get_probability(features_g)

            # INFO: Calculates norm gap
            outputs_norm_gap = calc_norm(outputs_g - outputs_l, logit=True, p=norm)
            features_norm_gap = calc_norm(features_g - features_l, p=norm)

            outputs_norm_gap_list.append(torch.mean(outputs_norm_gap))
            features_norm_gap_list.append(sum_mean(features_norm_gap))

    outputs_gap = torch.Tensor(outputs_norm_gap_list)
    features_gap = torch.Tensor(features_norm_gap_list)

    if prob:
        summary_writer.add_scalar("norm_gap/l{}-probability/feature".format(norm), torch.mean(outputs_gap), epoch)
        summary_writer.add_scalar("norm_gap/l{}-probability/classifier".format(norm), torch.mean(features_gap), epoch)
    else:
        summary_writer.add_scalar("norm_gap/l{}/feature".format(norm), torch.mean(outputs_gap), epoch)
        summary_writer.add_scalar("norm_gap/l{}/classifier".format(norm), torch.mean(features_gap), epoch)


# For hessian matrix value
# Hessian matrix is related loss-landscape convexity
# Warning : calculating hessian value is very time consuming task.
# def mark_hessian(model: Module, data_loader: DataLoader, summary_writer: SummaryWriter, epoch: int) -> None:
#     """
#     Compute the accuracy using its whole data.
#
#     Args:
#         model: (torch.Module) Training model.
#         data_loader: (torch.utils.Dataloader) Dataloader.
#         summary_writer: (SummaryWriter) SummaryWriter object.
#         epoch: (int) Current global round.
#
#     Returns: ((float) maximum eigenvalue of hessian, (float) hessian trace)
#
#     """
#     device = "cuda" if torch.cuda.is_available() is True else "cpu"
#
#     model.to(device)
#     model.eval()
#
#     hessian_trace = 0.0
#     max_eigval = 0.0
#     count = 0
#
#     loss_fn = torch.nn.CrossEntropyLoss().to(device)
#
#     for x, y in data_loader:
#         x = x.to(device)
#         y = y.to(device).to(torch.long)
#         outputs, _ = model(x)
#         if loss_fn is not None:
#             loss = loss_fn(outputs, y)
#
#             grad1 = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#
#             grad1 = torch.cat([grad.flatten() for grad in grad1])
#
#             count += len(grad1)
#
#             for i in range(grad1.size(0)):
#
#                 grad2 = torch.autograd.grad(grad1[i], model.parameters(), create_graph=True)
#
#                 grad2 = torch.cat([grad.flatten() for grad in grad2])
#
#                 hessian_trace += grad2.sum().item()
#
#                 max_eigval = max(max_eigval, torch.abs(grad2).max().item())
#         break
#
#     hessian_trace /= count
#     max_eigval /= count
#
#     summary_writer.add_scalar("max_hessian_eigen",max_eigval,epoch)
#     summary_writer.add_scalar("hessian_trace",hessian_trace,epoch)

def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids

def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)




def mark_hessian(model: Module, data_loader: DataLoader, summary_writer: SummaryWriter, epoch: int) -> None:
    """
    Compute the accuracy using its whole data.

    Args:
        model: (torch.Module) Training model.
        data_loader: (torch.utils.Dataloader) Dataloader.
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.

    Returns: ((float) maximum eigenvalue of hessian, (float) hessian trace)

    """
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    model.to(device)
    model.eval()

    hessian_trace = 0.0
    max_eigval = 0.0
    count = 0

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    hessian_comp = hessian(model, loss_fn, dataloader=data_loader, cuda=True) # use it for computing hessian
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=5)

    trace = hessian_comp.trace()
    density_eigens, density_weights = hessian_comp.density()

    density, grids = density_generate(density_eigens, density_weights)

    fig = plt.figure()

    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([np.min(density_eigens) - 1, np.max(density_eigens) + 1, None, None])
    plt.tight_layout()

    # have to save the figure...
    del hessian_comp

    lambda_min = np.min(density_eigens)
    lambda_max = np.max(density_eigens)

    lambda_maxratio = top_eigenvalues[0]/ top_eigenvalues[4]

    lambda_ratio = lambda_max/lambda_min## higher, better
    if lambda_ratio < 0.0:
        lambda_ratio = 0.0-lambda_ratio


    summary_writer.add_scalar("max_hessian_eigen",lambda_max,epoch)
    summary_writer.add_scalar("hessian_trace",trace,epoch)

    summary_writer.add_scalar("min_hessian_eigen",lambda_min,epoch)
    summary_writer.add_scalar("min_max_eigen_ratio",lambda_ratio,epoch)

    summary_writer.add_scalar("max_eigen/top5_eigen", lambda_maxratio, epoch)

    summary_writer.add_figure("Hessian_eigen_density/{}".format(epoch), fig)




#for cosine similarity check
def mark_cosine_similarity(current_state: OrderedDict, original_state: OrderedDict, summary_writer: SummaryWriter, epoch: int ) -> None:
    """
    Mark the cosine similarity between client and global
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.

    Returns: (None)

    """
    device = "cpu"

    original_params = []  ## flattened global_weight
    local_params = []     ## flattened client_weight


    for k in current_state.keys():
        original_params.append(torch.flatten(original_state[k].to(torch.float32)))
        local_params.append(torch.flatten(current_state[k].to(torch.float32)))

    ## flatten parameters and change to vector
    original_params = torch.cat(original_params)
    local_params = torch.cat(local_params)


    # INFO: Calculate Total Weight Similarity on each client
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    siml = cos_sim(local_params.to(device), original_params.to(device)).item()

    summary_writer.add_scalar("cosine_similarity", siml, epoch)


#for cosine similarity check
def mark_norm_size(current_state: OrderedDict, summary_writer: SummaryWriter, epoch: int ) -> None:
    """
    Mark the cosine similarity between client and global
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.

    Returns: (None)

    """
    #device = "cpu"

    local_params = []     ## flattened client_weight


    for k in current_state.keys():
        local_params.append(torch.flatten(current_state[k].to(torch.float32)))

    ## flatten parameters and change to vector
    local_params = torch.cat(local_params)

    # INFO: Calculate Total Weight norm size on each client
    norm_size = torch.norm(local_params, dim=-1)

    summary_writer.add_scalar("Norm_size", norm_size, epoch)


# for weight distribution visualization
# extract 1000 paramter and check distribution
# should be checked later
def mark_weight_distribution(clients, original_state: OrderedDict, summary_writer: SummaryWriter, epoch: int ) -> None:
    models_weights = []
    gmodel_weights = []

    for client in clients:
        model_weight = []
        ## collect each client paramters. sampling 200 of them on each
        for k in original_state.keys():
            model_weight.append(torch.flatten(client.model[k].to(torch.float32)))
        cweight = torch.cat(model_weight)

        indices = np.random.choice(cweight.shape[0], size=200, replace=False)  #extract random 200 element

        models_weight = cweight[indices]
        models_weights.append(torch.cat(model_weight))

    #collect global model paramter. sampling 1000
    gmodel_weights = []
    for k in original_state.keys():
        gmodel_weights.append(torch.flatten(original_state[k].to(torch.float32)))
    gmodel_weights=torch.cat(gmodel_weights)

    indices = np.random.choice(gmodel_weights.shape[0], size=200, replace=False)
    gmodel_weights = gmodel_weights[indices]

    ## draw density plot
    fig, axe = plt.subplots(1, 1, figsize=(8, 8))
    for i, weights in enumerate(models_weights):
        sns.kdeplot(data=weights, label="Client {}".format(i), ax=axe)
    sns.kdeplot(data=gmodel_weights, label="Global Model", ax=axe, linewidth=3, linestyle='--')

    axe.set_title("Weight Density Plot")
    axe.set_xlabel("Value")
    axe.set_ylabel("Density")
    axe.legend()

    summary_writer.add_figure("Weight Density Plots/{}".format(epoch), fig)



# TODO: Need to check it is surely works.
def kl_indicator(local_tensor, global_tensor, logit=False, alpha=1):
    # INFO: Calculates entropy gap
    entr_gap = calculate_entropy_gap(local_tensor, global_tensor, logit=logit)

    # INFO: Calculates norm gap
    l1_norm_gap = calculate_norm_gap(local_tensor, global_tensor, logit=logit)

    if logit:
        entr_gap = torch.mean(entr_gap)
        l1_norm_gap = torch.mean(l1_norm_gap)
    else:
        # Mean of feature maps and then batch.
        entr_gap = mean_mean(entr_gap)
        l1_norm_gap = mean_mean(l1_norm_gap)

    indicator = torch.sqrt(l1_norm_gap / (1 + alpha * entr_gap)).detach()
    return torch.nan_to_num(indicator)


# TODO: Need to be fixed.
# New Modification 22.07.20
def local_training_moon(clients: dict) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}

    Returns: None

    """
    ray.get([client.train_moon.remote() for _, client in clients.items()])


## INFO:
## 1.Centering Gradient  2.Orthogonalize Gradient
def Constrainting(original_state, current_state):
    """
    Mark the cosine similarity between client and global
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state

    Returns: new_state: (OrderedDict) Adjusted Model state
    """

    original_params = []  ## flattened glob_weight
    local_params = []  ## flattened updated_local_weight
    grad_state = OrderedDict()
    new_state = OrderedDict()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    for k in current_state.keys():
        current_state[k]=current_state[k].to(device)#
        original_state[k]=original_state[k].to(device)#
        original_params.append(torch.flatten(original_state[k].to(torch.float32)))
        local_params.append(torch.flatten(current_state[k].to(torch.float32)))
        grad_state[k] = current_state[k] - original_state[k]

    ## flatten parameters and change to vector
    original_params = torch.cat(original_params)
    local_params = torch.cat(local_params)
    grad_params = local_params - original_params


    #For centering & orthogonalizing gradient
    grad_mean = torch.mean(grad_params)

    gmean_params = torch.full(local_params.shape, grad_mean)

    grad_params -= gmean_params.to(device)

    GG = torch.dot(original_params, original_params)
    G = torch.sqrt(GG)

    dGG = torch.dot(grad_params, grad_params)
    dG = torch.sqrt(dGG)

    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    C = cos_sim(grad_params, original_params)

    parallel_scale = C * dG / G
    ################################################################################

    ## INFO: Update the local model with centered & orthogonalized gradient
    for k in current_state.keys():
        gmean_tensor = torch.full(grad_state[k].shape, grad_mean)
        new_state[k] = original_state[k] - original_state[k] * parallel_scale + grad_state[k] - gmean_tensor.to(device)

    return new_state

## INFO:
## 1.Centering Gradient  2.Orthogonalize Gradient

def Constrainting_layer_per_layer(original_state, current_state):
    """
    Adjust the gradient for each layer using centering and orthogonalization.
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state

    Returns: new_state: (OrderedDict) Adjusted Model state
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    new_state = OrderedDict()

    for k in current_state.keys():
        # Normalize global parameters using L2 norm to satisfy w^2 = 2.0
        # ori_mean = torch.mean(original_state[k])
        # original_state[k] = original_state[k] - ori_mean

        #l2_norm = torch.norm(original_state[k])
        #scaling_factor = torch.sqrt(torch.tensor(2.0)) / l2_norm
        #original_state[k] = original_state[k] * scaling_factor

        current_state[k]=current_state[k].to(device)
        original_state[k]=original_state[k].to(device)

        grad = current_state[k] - original_state[k]

        # Centering the gradient
        grad_mean = torch.mean(grad)
        grad_centered = grad - grad_mean

        # Calculating cosine similarity for orthogonalization
        C = torch.nn.CosineSimilarity(dim=-1)
        cos_sim = C(grad_centered.flatten(), original_state[k].flatten())

        # Calculating the scales for orthogonalization
        G = torch.norm(original_state[k])
        dG = torch.norm(grad_centered)
        parallel_scale = cos_sim * dG / G

        # Orthogonalizing the gradient
        grad_orthogonalized = grad_centered - (parallel_scale * original_state[k].flatten()).reshape_as(grad)

        # Update the local model with centered & orthogonalized gradient
        new_state[k] = original_state[k] + grad_orthogonalized

    return new_state

## scale to proportional to weight on allocated perturbation
## 0.01 ~ 0.20
# loss_fn: Optional[Module] = None
def compute_loss_along_random_direction(model, alpha, criterion, data_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 시드 설정
    random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    # 무작위 방향 생성
    direction = {name: torch.randn_like(param).to(device) for name, param in model.named_parameters()}
    normalized_direction = {name: d / (d.norm() + 1e-5) for name, d in direction.items()}

    original_state = {name: param.clone() for name, param in model.named_parameters()}

    # perturb model
    for name, param in model.named_parameters():
        norm = param.data.norm()
        param.data.add_(alpha * normalized_direction[name]*norm)

    # loss 계산
    total_loss = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device).to(torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    # 원래 모델 상태로 복원
    model.load_state_dict(original_state)
    return total_loss / len(data_loader)

def compute_loss_slope(model: Module, data_loader: DataLoader,summary_writer: SummaryWriter, epoch: int,  loss_fn: Optional[Module] = None):
    # loss 계산
    alphas = np.linspace(-1, 1, 100)
    losses = [compute_loss_along_random_direction(model, alpha, loss_fn, data_loader) for alpha in alphas]

    # 계산된 loss를 시각화
    fig = plt.figure()
    plt.plot(alphas, losses)
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Along a Random Direction")
    plt.show()
    summary_writer.add_figure("Weight Density Plots/{}".format(epoch), fig)


def compute_feature_weight_stat(model: Module,
                                data_loader: DataLoader,
                                seed: int = 42) -> Tuple[float, float, float, float]:
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 시드 설정
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 모델의 초기 상태 저장
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 무작위로 하나의 layer 선택
    param_keys = list(model.state_dict().keys())
    chosen_key = random.choice(param_keys)

    # '.weight' 또는 '.bias'를 제거하여 layer_key를 얻습니다.
    layer_key = chosen_key.rsplit('.', 1)[0]

    # layer의 feature 차원 확인
    if ".weight" in chosen_key or ".bias" in chosen_key:
        feature_dim = model.state_dict()[chosen_key].shape[0]
    else:
        raise ValueError(f"Unsupported parameter key: {chosen_key}")

    # 무작위로 하나의 feature 선택
    feature_index = random.randint(0, feature_dim - 1)

    model.to(device)
    model.eval()

    beta = 0.99  # Moving average factor
    moving_avg_mean = None
    moving_avg_var = None

    def hook(module, input, output):
        nonlocal moving_avg_mean, moving_avg_var

        #feature_output = input[0][:, feature_index] ## use input value for bn layer
        feature_output = output[:, feature_index] ## use output value

        batch_mean = torch.mean(feature_output).detach()
        batch_var = torch.var(feature_output).detach()

        if moving_avg_mean is None:
            moving_avg_mean = batch_mean
            moving_avg_var = batch_var
        else:
            moving_avg_mean = beta * moving_avg_mean + (1 - beta) * batch_mean
            moving_avg_var = beta * moving_avg_var + (1 - beta) * batch_var

    # 특정 layer에 hook 추가
    layer = dict(model.named_modules())[layer_key]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            outputs = model(x)

    # Hook 제거
    handle.remove()

    # 해당 feature의 weight 값의 평균과 분산 계산
    weight_values = model.state_dict()[chosen_key][feature_index]
    weight_mean = torch.mean(weight_values).item()
    weight_var = torch.var(weight_values).item()

    # 모델의 상태를 원래대로 복구 # 필요한지??
    model.load_state_dict(original_state)

    return moving_avg_mean.item(), moving_avg_var.item(), weight_mean, weight_var


def Normalize(original_state):
    """
    Adjust the gradient for each layer using centering and orthogonalization.
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state

    Returns: new_state: (OrderedDict) Adjusted Model state
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for k in original_state.keys():
        original_state[k] = original_state[k].to(device)
        if original_state[k].dim() == 4:
            # Normalize global parameters using L2 norm to satisfy w^2 = 2.0
            # CNN 레이어에 대한 연산
            ori_mean = original_state[k].mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            original_state[k] = original_state[k] - ori_mean

            l2_norm = original_state[k].norm(dim=1, keepdim=True).norm(dim=2, keepdim=True).norm(dim=3, keepdim=True)
            scaling_factor = torch.sqrt(torch.tensor(2.0)) / l2_norm
            original_state[k] = original_state[k]*scaling_factor

        elif original_state[k].dim() == 2:
            # Normalize global parameters using L2 norm to satisfy w^2 = 2.0
            # Linear 레이어에 대한 연산
            ori_mean = torch.mean(original_state[k], dim=1, keepdim=True)
            original_state[k] = original_state[k] - ori_mean

            l2_norm = torch.norm(original_state[k], dim=1, keepdim=True)
            scaling_factor = torch.sqrt(torch.tensor(2.0)) / l2_norm
            original_state[k] = original_state[k]*scaling_factor

        else:
            original_state[k] = original_state[k]

    return original_state

## INFO:
## 1.Centering Gradient  2.Orthogonalize Gradient

def Constrainting_strict(original_state, current_state):
    """
    Adjust the gradient for each layer using centering and orthogonalization.
    Args:
        current_state: (OrderedDict) Local Model state
        original_state: (OrderedDict) Global Model state

    Returns: new_state: (OrderedDict) Adjusted Model state
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    new_state = OrderedDict()
    C = torch.nn.CosineSimilarity(dim=-1)

    for k in current_state.keys():

        current_state[k]=current_state[k].to(device)
        original_state[k]=original_state[k].to(device)
        grad = current_state[k] - original_state[k]

        if grad.dim() == 4:  # CNN layer
            # CNN 레이어에 대한 연산
            G = original_state[k].norm(dim=1, keepdim=True).norm(dim=2, keepdim=True).norm(dim=3, keepdim=True)
            dG = grad.norm(dim=1, keepdim=True).norm(dim=2, keepdim=True).norm(dim=3, keepdim=True)

            gmean_tensor = grad.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            grad -= gmean_tensor

            first, _, __, __ = original_state[k].size()
            Gl = original_state[k].view(first, -1)
            dGl = grad.view(first, -1)

            cos_sim = C(Gl, dGl)

            cos_sim = cos_sim[:, None, None, None]
            parallel_scale = (cos_sim * dG) / G
            new_state[k] = original_state[k] + grad - (parallel_scale * original_state[k])
        elif grad.dim() == 2:  # Linear layer
            # Linear 레이어에 대한 연산
            G = torch.norm(original_state[k], dim=1, keepdim=True)
            dG = torch.norm(grad, dim=1, keepdim=True)

            gmean_tensor = torch.mean(grad, dim=1, keepdim=True)
            grad -= gmean_tensor

            cos_sim = C(grad, original_state[k])
            cos_sim = cos_sim[:, None]
            parallel_scale = (cos_sim * dG) / G
            new_state[k] = original_state[k] + grad - (parallel_scale * original_state[k])
        # elif grad.dim() == 2:  # Linear layer
        #     # Linear 레이어에 대한 연산
        #     G = torch.norm(original_state[k], dim=1, keepdim=True).norm(dim=2, keepdim=True)
        #     dG = torch.norm(grad, dim=1, keepdim=True).norm(dim=2, keepdim=True)
        #     gmean_tensor = torch.mean(grad, dim=1, keepdim=True).mean(dim=2, keepdim=True)
        #     grad -= gmean_tensor
        #
        #     cos_sim = C(grad, original_state[k])
        #     cos_sim = cos_sim[:, None]
        #     parallel_scale = (cos_sim * dG) / G
        #     new_state[k] = original_state[k] + grad - (parallel_scale * original_state[k])

        else:
            new_state[k] = current_state[k]
            continue

    return new_state


## INFO:
## extracting model state_dict(paramters)
##
def get_parameters(model, ordict: bool = True) -> Union[OrderedDict, Any]:
    if ordict:
        return OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    else:
        return [val.clone().detach().cpu() for _, val in model.state_dict().items()]


## Sharpness Aware Minimization
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=1.0, adaptive = True, **kwargs): ## 0.5 // False
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups