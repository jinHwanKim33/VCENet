import json
import pickle
import random
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--scene_num', type=str, default="224", help="episode scene type")
parser.add_argument('--target', type=int, default=0, help="episode scene type")
args = parser.parse_args()

obj_catogory = ['AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
                'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
                'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster']

split_info = {'scene': 'FloorPlan228', 'goal_object_type': 'LightSwitch', 'task_data': ['LightSwitch|+00.00|+01.29|+03.38'], 'state': '-4.50|4.50|0|30'}

splits = []

for i in range(1000):
    split_info = {}

    object_file = "/media/ailab/afbeb78c-82c2-4640-be3f-67053d1554f9/pl_ws/DETR/AI2Thor_offline_data_2.0.2/FloorPlan" + args.scene_num + "/visible_object_map_1.5.json"
    with open(object_file, 'r') as f:
        object_data = json.load(f)

## !! caution - no check for existing
    task_data = []
    for keys in object_data.keys():
        if obj_catogory[args.target] == keys.split("|")[0]:
            task_data.append(keys)

    states = []
    state_file = "/media/ailab8503/484d268c-c692-47d6-800f-b6c2d2f92790/junghyeon_workspace/Data/AI2Thor_offline_data_2.0.2/FloorPlan" + args.scene_num + "/graph.json"
    with open(state_file, 'r') as f:
        state_data = json.load(f)

    for j in range(len(state_data['nodes'])):
        states.append(state_data['nodes'][j]['id'])
    state = random.choice(states)

    split_info['scene'] = "FloorPlan"+args.scene_num
    split_info['goal_object_type'] = obj_catogory[args.target]
    split_info['task_data'] = task_data
    split_info['state'] = state

    splits.append(split_info)

with open('one_FloorPlan'+args.scene_num+'_train_22.pkl', 'wb') as f:
    pickle.dump(splits, f)