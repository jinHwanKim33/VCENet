import json
import pickle
import random
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--episode_type', type=str, default="kitchen", help="episode scene type")
args = parser.parse_args()

obj_catogory = ['AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
                'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
                'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster']

splits = []

for _ in range(1000):
    scenes = []
    split_info = {}
    num = 0

    if args.episode_type == "kitchen":
        num = 0
    elif args.episode_type == "living_room":
        num = 1
    elif args.episode_type == "bedroom":
        num = 2
    elif args.episode_type == "bathroom":
        num = 3

    for j in range(20):
        if num == 0:
            scenes.append("FloorPlan" + str(j + 1))
        else:
            scenes.append("FloorPlan" + str(num + 1) + '%02d' % (j + 1))
    scene = random.choice(scenes)

    object_file = "/media/ailab8503/484d268c-c692-47d6-800f-b6c2d2f92790/junghyeon_workspace/Data/AI2Thor_offline_data_2.0.2/" + scene + "/visible_object_map_1.5.json"
    state_file = "/media/ailab8503/484d268c-c692-47d6-800f-b6c2d2f92790/junghyeon_workspace/Data/AI2Thor_offline_data_2.0.2/" + scene + "/graph.json"

    with open(object_file, 'r') as f:
        object_data = json.load(f)

    tasks = []
    task_data = []
    goal_obj_types = []
    for keys in object_data.keys():
        if keys.split("|")[0] in obj_catogory:
            tasks.append(keys)
            goal_obj_types.append(keys.split("|")[0])
    goal_obj_type = random.choice(goal_obj_types)

    for t in range(len(tasks)):
        if goal_obj_type in tasks[t]:
            task_data.append(tasks[t])


    states = []
    with open(state_file, 'r') as f:
        object_data = json.load(f)

    for i in range(len(object_data['nodes'])):
        states.append(object_data['nodes'][i]['id'])
    state = random.choice(states)


    split_info['scene'] = scene
    split_info['goal_object_type'] = goal_obj_type
    split_info['task_data'] = task_data
    split_info['state'] = state



    splits.append(split_info)

with open(args.episode_type + '_train_22.pkl', 'wb') as f:
    pickle.dump(splits, f, pickle.HIGHEST_PROTOCOL)
