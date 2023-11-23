from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

def update_test_model(args, player, target_action_prob, weight=1):
    action_loss = weight * F.cross_entropy(player.last_action_probs, torch.max(target_action_prob, 1)[1])
    inner_gradient = torch.autograd.grad(
        action_loss,
        [v.clone() for _, v in player.model.named_parameters()],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    player.model.load_state_dict(SGD_step(player.model, inner_gradient, args.inner_lr))
    player.episode.model_update = True


def run_episode(player, args, total_reward, model_options, training, shared_model=None):
    num_steps = args.num_steps
    update_test_type = {
        'meta_learning': args.update_test_meta_learning
    }

    update_test = True in update_test_type.values()

    for _ in range(num_steps):
        #player.action(model_options, training, update_test)
        ## up is default  ##hwan
        _, _, action = player.action(model_options, training, update_test)
        if update_test_type['meta_learning']:
            current_state = str(player.episode.environment.controller.state)
            if current_state in player.episode.states:  # and action==0:

                player.episode.deadlock_state.append(True)
                target_action_prob = player.episode.meta_predictions[-1]
                player.episode.tpn_actions.append(target_action_prob.argmax(dim=1, keepdim=True).tolist()[0][0])  ##detach??
                update_test_model(args, player, target_action_prob, 1)
            else:
                player.episode.deadlock_state.append(False)
                player.episode.tpn_actions.append("None")

            # if current_state in player.episode.states:
            #     player.episode.deadlock_state.append(True)
            # else:
            #     player.episode.deadlock_state.append(False)
        total_reward = total_reward + player.reward
        if player.done:
            break

    return total_reward


def new_episode(args, player):
    player.episode.new_episode(args, player.scenes, player.targets)
    player.reset_hidden()
    player.done = False


def a3c_loss(args, player, gpu_id, model_options):
    """ Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """
    R = torch.zeros(1, 1)
    if not player.done:
        _, output = player.eval_at_state(model_options)
        R = output.value.data  #critic
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    player.values.append(Variable(R))
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()
    R = Variable(R)
    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = (
                player.rewards[i]
                + args.gamma * player.values[i + 1].data
                - player.values[i].data
        )

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = (
                policy_loss
                - player.log_probs[i] * Variable(gae)
                - args.beta * player.entropies[i]
        )

    return policy_loss, value_loss


## choi
# def imitation_learning_loss(args, player):
#     episode_loss = torch.tensor(0)
#     with torch.cuda.device(player.gpu_id):
#         episode_loss = episode_loss.cuda()
#
#     for i in player.il_update_actions:
#         a = list(player.il_update_actions.keys())[0]
#         step_optimal_action = torch.tensor(player.il_update_actions[i]).reshape([1]).long()
#         with torch.cuda.device(player.gpu_id):
#             step_optimal_action = step_optimal_action.cuda()
#         step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
#         episode_loss = episode_loss + step_loss * math.pow(args.il_gamma, i - a + 1)
#
#     return episode_loss

# def meta_learning_loss(player):
#     episode_loss = torch.tensor(0)
#     with torch.cuda.device(player.gpu_id):
#         episode_loss = episode_loss.cuda()
#     for i in player.meta_learning_actions:
#         step_optimal_action = torch.tensor(player.meta_learning_actions[i]).reshape([1]).long()
#         with torch.cuda.device(player.gpu_id):
#             step_optimal_action = step_optimal_action.cuda()
#         step_loss = F.cross_entropy(player.meta_predictions[i], step_optimal_action)
#         #step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
#         episode_loss = episode_loss + step_loss
#     return episode_loss

def imitation_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.il_update_actions:
        step_optimal_action = torch.tensor(player.il_update_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss


def meta_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.meta_learning_actions:
        step_optimal_action = torch.tensor(player.meta_learning_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.meta_predictions[i], step_optimal_action)
        #step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss
    return episode_loss

# n step + gamma - server2
# def meta_learning_loss(args, player):
#     episode_loss = torch.tensor(0)
#     with torch.cuda.device(player.gpu_id):
#         episode_loss = episode_loss.cuda()
#     if list(player.meta_learning_actions.keys()):
#         s_step = list(player.meta_learning_actions.keys())[0]  # start
#     else:
#         s_step = 0  # start
#     p_step, c_step = 0, 0  # previous, current
#
#     for i in player.meta_learning_actions:
#         c_step = i
#         if c_step - 1 != p_step:
#             s_step = c_step
#
#         step_optimal_action = torch.tensor(player.meta_learning_actions[i]).reshape([1]).long()
#         with torch.cuda.device(player.gpu_id):
#             step_optimal_action = step_optimal_action.cuda()
#         step_loss = F.cross_entropy(player.meta_predictions[i], step_optimal_action)
#         # step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
#         episode_loss = episode_loss + step_loss * math.pow(args.il_gamma, c_step - s_step)
#         # print(math.pow(args.il_gamma, c_step - s_step))
#         p_step = c_step
#
#     return episode_loss


def duplicate_states_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.duplicate_states_actions:
        step_optimal_action = torch.tensor(player.duplicate_states_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss


def compute_learned_loss(args, player, gpu_id, model_options):
    loss_hx = torch.cat((player.hidden[0], player.last_action_probs), dim=1)
    learned_loss = {
        "learned_loss": player.model.learned_loss(
            loss_hx, player.learned_input, model_options.params
        )
    }
    player.learned_input = None
    return learned_loss


def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
            player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            # print("player_param_grad ; ", param.grad)
            # print("shared_parma_grad ; ", shared_param.grad)
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()


def transfer_gradient_to_shared(gradient, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape)
            elif gpu_id < 0:
                param._grad = gradient[i]
            else:
                param._grad = gradient[i].cpu()

        i += 1


def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            # theta[name] = torch.tensor(
            #     param_copied,
            #     requires_grad=True,
            #     device=torch.device("cuda:{}".format(gpu_id)),
            # )
            # Changed for pythorch 0.4.1.
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta


def update_loss(sum_total_loss, total_loss):
    if sum_total_loss is None:
        return total_loss
    else:
        return sum_total_loss + total_loss


def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()


def SGD_step(theta, grad, lr):
    theta_i = {}
    j = 0
    for name, param in theta.named_parameters():
        #print("name : ", name, param)
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param
        j += 1

        #print("name : ", name, torch.equal(param,theta_i[name]))
    #print("TPN_Learned_theta : ", theta_i)
    return theta_i


def get_scenes_to_use(player, scenes, args):
    if args.new_scene:
        return scenes
    return [player.episode.environment.scene_name]


def compute_loss(args, player, gpu_id, model_options):
    if not args.only_il_loss:

        loss = {'policy_loss': a3c_loss(args, player, gpu_id, model_options)[0],
                'value_loss': a3c_loss(args, player, gpu_id, model_options)[1]}
        loss['total_loss'] = loss['policy_loss'] + 0.5 * loss['value_loss']
        if args.imitation_learning:
            loss['il_loss'] = imitation_learning_loss(player)  # args,
            loss['total_loss'] = loss['total_loss'] + args.il_rate * loss['il_loss']
        elif args.memory_duplicate_learning:
            loss['memory_loss'] = duplicate_states_loss(player)
            loss['total_loss'] = loss['total_loss'] + args.memory_duplicate_rate * loss['memory_loss']
    elif args.update_meta_network and args.only_il_loss:
        meta_loss = meta_learning_loss(player)   # args,
        loss = {'meta_loss': meta_loss,
                'total_loss': meta_loss}

    return loss


def end_episode(
        player, res_queue, title=None, episode_num=0, include_obj_success=False, **kwargs
):
    actions_list = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Done']
    actions_ = []
    # optimal_actions = []
    # optimal_actions.append(player.environment.controller.get_optimal_action(player.episode.task_data[0].split("|")[0]))
    # print("0", player.actions)
    for item in player.actions:
        item = item.tolist()
        # print(item[0][0])
        # print("1", item. int(item[0][0]), actions_list[int(item[0][0])])
        actions_.append(actions_list[item[0][0]])
    # print(actions_)
    # for item in player.episode.optimal_actions_record:
    #     optimal_actions.append(action_list[int(item)])
    results = {
        'done_count': player.episode.done_count,
        'ep_length': player.eps_len,
        'success': int(player.success),
        'tools': {
            'scene': player.episode.scene,
            'target': player.episode.task_data[0].split("|")[0],
            'states': player.episode.states,
            'deadlock_state': player.episode.deadlock_state,
             #'action_outputs': player.episode.action_outputs,
            'action_list': actions_,
            'tpn_action_list': player.episode.tpn_actions,
            # 'optimal_action_list' : optimal_actions,
            'detection_results': player.episode.detection_results,
            'success': player.success,
            'match_score': player.episode.match_score,
            #'indices_topk': player.episode.indices_topk_list,
            'y': player.episode.environment.controller.state.y,


            # 'graph_info': player.episode.graph_info_list,
            # 'action_list': [int(item) for item in player.episode.actions_record],
            # 'reward': player.reward,
            # 'all_object_results': player.episode.all_object_bbx,
        }
    }

    results.update(**kwargs)
    res_queue.put(results)


def get_bucketed_metrics(spl, best_path_length, success):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
    return out


def compute_spl(player, start_state):
    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            continue

    if not player.success:
        return 0, best

    if best < float("inf"):
        return best / float(player.eps_len), best

    return 0, best


def action_prob_detection(bbox):
    center_point = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    left_prob = np.linalg.norm(center_point - np.array([0, 150]))
    right_prob = np.linalg.norm(center_point - np.array([300, 150]))
    up_prob = np.linalg.norm(center_point - np.array([150, 0]))
    down_prob = np.linalg.norm(center_point - np.array([150, 300]))
    forward_prob = np.linalg.norm(center_point - np.array([150, 150]))

    detection_prob = torch.tensor([forward_prob, left_prob, right_prob, up_prob, down_prob])

    return torch.argmin(detection_prob)
