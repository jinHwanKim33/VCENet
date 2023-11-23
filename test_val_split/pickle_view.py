#!/usr/bin/env python


import pickle
import sys

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) <= 1:
        print('Specify pickle file as parameter.')
    else:
        file = pickle.load(open(argv[1], 'rb'))
        print(file)
        # for i in range(len(file)):
        #     if '0.25|2.75|0|30' in file[i]['state']:
        #         print(file[i])
        # # print(pickle.load(open(argv[1], 'rb')))
        # # list = []
        # for i in a:
        #     # list.append(i['scene'])
        #    print(i)
        # # print(len(list))
        # print(type(a))
#python pickle_view.py ~~~
