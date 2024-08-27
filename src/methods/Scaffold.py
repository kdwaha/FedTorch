from . import *
from src.model import NUMBER_OF_CLASSES
from .utils import *


@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int):
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes,training_settings['bn'])
    model.load_state_dict(client.model) ## load state dict, not all attribute.
    ## it's own


    model = model.to(device)
    original_state = F.get_parameters(model)
    if training_settings['localrie']:
        original_state = F.Constrainting_sphere(original_state)
        model.load_state_dict(original_state, strict=True)

    #Check if there exist correction term.

    if not hasattr(client,'correction'):
        client.correction = {}
        for k, v in original_state.items():
            client.correction[k] = torch.zeros_like(v.data)

    if not hasattr(client,'gcorrection'):
        client.gcorrection = client.correction

    # INFO - Optimizer
    optimizer = call_optimizer(training_settings['optim'])

    # INFO - Optimizations
    if training_settings['optim'].lower() == 'sgd':
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'],
                          momentum=training_settings['momentum'], weight_decay=training_settings['weight_decay'])
    else:
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'])

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    velocity = {}
    gmean = {}
    gmean_prev = {}
    gmeanrie = {}
    gmean_prevrie = {}
    gvar = 0
    gnorm = 0
    gvarrie = 0
    gnormrie = 0
    gsq = {}
    gsqrie = {}
    gvarriev = {}
    gvarv = {}

    for k, v in original_state.items():
        gmean[k] = torch.zeros_like(v.data)
        gmean_prev[k] = torch.zeros_like(v.data)
        gmeanrie[k] = torch.zeros_like(v.data)
        gmean_prevrie[k] = torch.zeros_like(v.data)

        gsq[k] = torch.zeros_like(v.data)
        gsqrie[k] = torch.zeros_like(v.data)
        gvarv[k] = torch.zeros_like(v.data)
        gvarriev[k] = torch.zeros_like(v.data)
        velocity[k] = torch.zeros_like(v.data)
    if training_settings['localrie']:  ##  일단 보류
        client.correction = F.Constrainting_grad(original_state, client.correction)
        client.gcorrection = F.Constrainting_grad(original_state, client.gcorrection)



    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        training_loss = 0
        summary_counter = 0

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device).to(torch.long)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)


            prev_state = F.get_parameters(model) #state befor update

            loss.backward()
            current_state = F.get_parameters(model)


            for k in current_state.keys():
                prev_state[k]  = prev_state[k].to(device)
                current_state[k] =current_state[k].to(device)
                original_state[k] = original_state[k].to(device)

            # applying correction mechanism of scaffold
            for k, v in model.named_parameters():
                #grad = (prev_state[k] - current_state[k]) / training_settings['local_lr'] ###
                if v in optim.state and 'momentum_buffer' in optim.state[v]:
                    velocity[k] = optim.state[v]['momentum_buffer']  ## copy momentum buffer
                else:
                    velocity[k] = v.grad.data


                # con = client.correction[k].clone().detach().cpu()
                # gcon = client.gcorrection[k].clone().detach().cpu()

                con = client.correction[k].clone().detach().to(device)
                gcon = client.gcorrection[k].clone().detach().to(device)

                #dp = grad + gcon - con
                velocity[k] = velocity[k] + gcon -con
                if v in optim.state and 'momentum_buffer' in optim.state[v]:
                    optim.state[v]['momentum_buffer'] = velocity[k]  ## apply adjust momentum
                else:
                    v.grad.data = velocity[k]

                #current_state[k] -= dp * training_settings['local_lr']
            if not training_settings['localrie']:
                optim.step()
            ############## For constraint  ###############################
            #current_state = F.Constrainting(original_state,current_state)
            if training_settings['const']:
                #current_state = F.Constrainting_layer_per_layer(original_state, current_state)
                #current_state = F.Constrainting_strict(original_state, current_state)
                current_state = F.Constrainting_sphere(current_state)
                model.load_state_dict(current_state, strict=True)

            ##############################################################

            # for k in current_state.keys():
            #     current_state[k] = current_state[k].to(device)
            #     prev_state[k] = prev_state[k].to(device)
            #     gmean[k] =gmean[k].to(device)
            #
            #     gnorm += ((current_state[k] - prev_state[k]).norm(2)**2).detach()  # accumulating
            #     gmean[k] += (current_state[k] - prev_state[k]).detach()  # accumulating mean , not for record
            #     gvar += ((current_state[k] - prev_state[k] - gmean_prev[k].to(device)).norm(2)**2).detach()  ## this way
            #

            ## for printing metric
            gvar = 0.0
            gnorm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Compute the correction term
                    gmean[name] = 0.99 * gmean[name].to(device)
                    gsq[name] = 0.99 * gsq[name].to(device)

                    gmean[name] += 0.01 * param.grad.to(device)
                    gsq[name] += 0.01 * (param.grad.to(device) * param.grad.to(device))

                    gvarv[name] = (gsq[name] - gmean[name] * gmean[name])
                    gvar = gvarv[name].norm(1).detach()
                    gnorm += (param.grad.norm(2) ** 2).detach()  ##.to(param.grad.device)



                #optim.step()
                #current_state = F.get_parameters(model)

            if training_settings['localrie']:

                #current_state = F.Constrainting_strict(prev_state, current_state) # asserting only orthogonal part to be remained

                for k, v in model.named_parameters():
                    if v in optim.state and 'momentum_buffer' in optim.state[v]:
                        velocity[k] = optim.state[v]['momentum_buffer']  ## copy momentum buffer
                    else:
                        velocity[k] = v.grad.data

                velocity = F.Constrainting_grad(prev_state, velocity)

                for k, v in model.named_parameters():
                    if v in optim.state and 'momentum_buffer' in optim.state[v]:
                        optim.state[v]['momentum_buffer'] = velocity[k] ## apply adjust momentum
                    else:
                        v.grad.data = velocity[k]

                optim.step()

                current_state = F.get_parameters(model)
                gvarrie = 0.0
                gnormrie = 0.0

                for k in current_state.keys():
                    current_state[k] = current_state[k].to(device)
                    prev_state[k] = prev_state[k].to(device)
                    gmeanrie[k] = gmeanrie[k].to(device)
                    gsqrie[k] = gsqrie[k].to(device)


                    gmean[k] = 0.99*gmeanrie[k].to(device)
                    gsqrie[k] = 0.99*gsqrie[k].to(device)

                    gmean[k] += (current_state[k]-prev_state[k]).detach()*0.01
                    gsqrie[k] += ((current_state[k]-prev_state[k])*(current_state[k]-prev_state[k])).detach()

                    gnormrie += ((current_state[k]-prev_state[k]).norm(2)**2).detach() #accumulating
                    #gmeanrie[k] += (current_state[k]-prev_state[k]).detach()*0.01 #accumulating mean , not for record
                    #gvarrie += ((gsqrie[k] - gmeanrie[k]*gmeanrie[k]).norm(2)**2).detach()# this way
                    gvarriev[k] = (gsqrie[k] - gmeanrie[name] * gmeanrie[name])
                    gvarrie = gvarriev[k].norm(1).detach()
                    #gnorm += (param.grad.norm(2) ** 2).detach()  ##.to(param.grad.device)


                #current_state = F.Exponential(prev_state,current_state)
                #model.load_state_dict(current_state, strict=True)



            model.load_state_dict(current_state, strict=True)

            # INFO - Step summary
            training_loss += loss.item()

            client.step_counter += 1
            summary_counter += 1

            if summary_counter % training_settings["summary_count"] == 0:
                training_acc, _ = F.compute_accuracy(model, client.train_loader, loss_fn)
                summary_writer.add_scalar('step_loss', training_loss / summary_counter, client.step_counter)
                summary_writer.add_scalar('step_acc', training_acc, client.step_counter)

                summary_writer.add_scalar('step_gnorm', gnorm, client.step_counter)
                summary_writer.add_scalar('step_gvar', gvar, client.step_counter)

                if training_settings['localrie']:
                    summary_writer.add_scalar('step_gnormrie', gnormrie, client.step_counter)
                    summary_writer.add_scalar('step_gvarrie', gvarrie, client.step_counter)

                gvar = 0
                gnorm = 0
                gvarrie = 0
                gnormrie = 0


                for k, v in original_state.items():
                    gmean_prev[k] = gmean[k] / summary_counter
                    gmean_prevrie[k] = gmeanrie[k] / summary_counter
                    gmean[k] = torch.zeros_like(v.data)
                    gmeanrie[k] = torch.zeros_like(v.data)

                summary_counter = 0
                training_loss = 0



        ##updating correction term
        #for k in current_state.keys():
        #    client.correction[k] = client.correction[k].to(device) - client.gcorrection[k].to(device)\
        #                                     + (original_state[k].to(device)-current_state[k].to(device))/ footprint

        # INFO - Epoch summary
        test_acc, test_loss = F.compute_accuracy(model, client.test_loader, loss_fn)
        train_acc, train_loss = F.compute_accuracy(model, client.train_loader, loss_fn)

        summary_writer.add_scalar('epoch_loss/train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('epoch_loss/test', test_loss, client.epoch_counter)

        summary_writer.add_scalar('epoch_acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('epoch_acc/local_test', test_acc, client.epoch_counter)

        # fmean, fvar, wmean, wvar = F.compute_feature_weight_stat(model, client.test_loader)
        # summary_writer.add_scalar('epoch_fmean/test', fmean, client.epoch_counter)
        # summary_writer.add_scalar('epoch_fvar/test', fvar, client.epoch_counter)
        # summary_writer.add_scalar('epoch_wmean/test', wmean, client.epoch_counter)
        # summary_writer.add_scalar('epoch_wvar/test', wvar, client.epoch_counter)

        ## Hessian info
        # F.mark_hessian(model, client.test_loader, summary_writer, client.epoch_counter)

        # F.mark_accuracy(client, model, summary_writer)
        # F.mark_entropy(client, model, summary_writer)

        F.mark_cosine_similarity(current_state, original_state, summary_writer, client.epoch_counter)
        F.mark_norm_size(current_state, summary_writer, client.epoch_counter)

        client.epoch_counter += 1
    e = training_settings['local_epochs']
    steps = e * len(client.train_loader)
    lr = training_settings['local_lr']
    footprint = steps * lr
    #current_state = F.Logarithm(original_state, current_state)
    #current_state = F.Constrainting_strict(original_state, current_state)
    for k in current_state.keys():
        client.correction[k] = client.correction[k].to(device) - client.gcorrection[k].to(device) \
                               + (original_state[k].to(device) - current_state[k].to(device)) / footprint


    # INFO - Local model update
    client.epoch_counter = client.epoch_counter
    client.step_counter = client.step_counter
    #current_state = F.get_parameters(model)
    #current_state = F.Logarithm(original_state, current_state)
    #model.load_state_dict(current_state, strict=True)
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    return client


def local_training(clients: list,
                   training_settings: dict,
                   num_of_class: int) -> list:
    """
    Args:
        clients: (dict) client ID and Object pair
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
    Returns: (List) Client Object result

    """
    # sampled_clients = random.sample(list(clients.values()), k=int(len(clients.keys()) * sample_ratio))
    ray_jobs = []
    for client in clients:
        if training_settings['use_gpu']:
            ray_jobs.append(train.options(num_gpus=training_settings['gpu_frac']).remote(client,
                                                                                         training_settings,
                                                                                         num_of_class))
        else:
            ray_jobs.append(train.options().remote(client,
                                                   training_settings,
                                                   num_of_class))
    trained_result = []
    while len(ray_jobs):
        done_id, ray_jobs = ray.wait(ray_jobs)
        trained_result.append(ray.get(done_id[0]))

    return trained_result


def fed_avg(clients: List[Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    total_len = 0
    empty_model = OrderedDict()
    empty_correction = OrderedDict()

    original_state = aggregator.get_parameters()

    for client in clients:
        total_len += client.data_len()

    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            if k not in empty_model.keys():
                empty_model[k] = client.model[k] * (1.0/len(clients)) * global_lr
                empty_correction[k] = client.correction[k] * (1.0/len(clients)) * global_lr
            else:
                empty_model[k] += client.model[k] * (1.0/len(clients)) * global_lr
                empty_correction[k] = client.correction[k] * (1.0/len(clients)) * global_lr
    #empty_model = F.Exponential(original_state, empty_model)

    # distribute aggregated correction-term
    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            client.gcorrection[k] = empty_correction[k]

    global_norm = 0
    gnorm2 = 0
    client_norm = 0
    for k, v in aggregator.model.state_dict().items():
        global_norm += ((empty_model[k]-original_state[k]).norm(2))**2
        gnorm2 += (empty_model[k].norm(2))**2
        for client in clients:
            client_norm += (((client.model[k] - original_state[k]).norm(2))**2) * (1.0 / len(clients))

    # Global model updates
    aggregator.set_parameters(empty_model)

    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()

    # TODO: Adapt in a future.
    # Calculate cos_similarity with previous representations
    # aggregator.calc_rep_similarity()
    #
    # # Calculate cos_similarity of weights
    # current_model = self.get_parameters()
    # self.calc_cos_similarity(original_model, current_model)
    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)
    aggregator.summary_writer.add_scalar('drift_diversity', client_norm / global_norm, aggregator.global_iter)
    aggregator.summary_writer.add_scalar('consistency', client_norm, aggregator.global_iter)
    F.mark_norm_size(empty_model, aggregator.summary_writer, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def run(client_setting: dict, training_setting: dict, b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = Client
    aggregator = Aggregator

    clients, aggregator = client_initialize(client, aggregator, fed_dataset, test_loader, valid_loader,
                                            client_setting, training_setting)
    original_state = aggregator.get_parameters()
    original_state = F.Constrainting_sphere(original_state)
    aggregator.set_parameters(original_state)

    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[3] Global step starts...")

        pbar = tqdm(range(training_setting['global_epochs']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        initial_lr = training_setting['local_lr']
        total_g_epochs = training_setting['global_epochs']

        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - Save the global model
            aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            stream_logger.debug("[*] Local training process...")
            # INFO - Normal Local Training
            sampled_clients = F.client_sampling(clients, sample_ratio=training_setting['sample_ratio'], global_round=gr)

            # INFO - COS decay
            # training_setting['local_lr'] = 1 / 2 * initial_lr * (
            #             1 + math.cos(aggregator.global_iter * math.pi / total_g_epochs))
            # stream_logger.debug("[*] Learning rate decay: {}".format(training_setting['local_lr']))
            # summary_logger.info("[{}/{}] Current local learning rate: {}".format(aggregator.global_iter,
            #                                                                      total_g_epochs,
            #                                                                      training_setting['local_lr']))

            trained_clients = local_training(clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])
            stream_logger.debug("[*] Federated aggregation scheme...")
            fed_avg(trained_clients, aggregator, training_setting['global_lr'])
            clients = F.update_client_dict(clients, trained_clients)

            # INFO - Save client models
            if b_save_model:
                save_model(clients)

            end_time_global_iter = time.time()
            pbar.set_postfix({'global_acc': aggregator.test_accuracy})
            summary_logger.info("Global Running time: {}::{:.2f}".format(gr,
                                                                         end_time_global_iter - start_time_global_iter))
            summary_logger.info("Test Accuracy: {}".format(aggregator.test_accuracy))
            if gr == training_setting['global_epochs'] - 1:
                F.compute_loss_slope(aggregator.model, aggregator.test_loader, aggregator.summary_writer, gr,
                                   torch.nn.CrossEntropyLoss())
            #if gr >= training_setting['global_epochs'] -3:
            #    F.mark_hessian(aggregator.model, aggregator.test_loader, aggregator.summary_writer, gr)
            # if gr % 10 == 0:
            #     F.mark_weight_distribution(trained_clients,aggregator.get_parameters(),aggregator.summary_writer,gr)
            #     F.mark_hessian(aggregator.model, aggregator.test_loader, aggregator.summary_writer,gr)

        summary_logger.info("Global iteration finished successfully.")
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    end_run_time = time.time()
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))

    # INFO - Save client's data
    if b_save_data:
        save_data(clients)
        aggregator.save_data()

    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")
