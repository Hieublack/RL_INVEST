from pickle import GLOBAL
from unittest import result
import pandas as pd
import math
import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import numba as nb


data_full = pd.read_csv('data_test.csv')
data_full = data_full.sort_values(by=['TIME', 'PROFIT'], ascending=[False, False], ignore_index=True)
all_fomula = np.array(pd.read_csv('congthuc.csv')['fomula'])
TOP_COMP_PER_QUARTER = 20
NUMBER_QUARTER_HISTORY = 24
ALL_QUARTER = len(np.unique(data_full['TIME']))

def get_index_T(data_full):
    list_T = data_full['TIME']
    index_T = [0]
    for i in range(len(list_T)-1):
        if list_T[i] != list_T[i+1]:
            index_T.append(i+1)
    index_T.append(len(list_T))
    return index_T
index_test = get_index_T(data_full)

def get_variable(data_full):
    global A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z, PROFIT, COMPANY
    list_variable = [[]]*26
    list_column = list(data_full.columns)
    PROFIT = np.array(data_full["PROFIT"])
    COMPANY = np.array(data_full["SYMBOL"])
    for i in range(4, len(list_column)):
        list_variable[i-4] = np.array(data_full[list_column[i]])
    [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z] = list_variable
get_variable(data_full)

def get_rank_not_invest():
    list_rank_ko_dau_tu = []
    for j in range(len(index_test)-1, 0, -1):
        profit_q = PROFIT[index_test[j-1]:index_test[j]]
        n_comp = len(profit_q)
        if np.max(profit_q) <= 1:
            list_rank_ko_dau_tu.append(1)
        elif np.min(profit_q) >= 1:
            list_rank_ko_dau_tu.append(n_comp + 1)
        else:
            list_rank_ko_dau_tu.append(np.where(profit_q <= 1)[0][0]+1)
    return np.array(list_rank_ko_dau_tu)
LIST_RANK_NOT_INVEST = get_rank_not_invest()
LIST_RANK_CT1 = []
LIST_RANK_CT2 = []

def get_in4_rank_fomula(fomula):
    result_ =  np.nan_to_num(eval(fomula), nan=-math.inf, posinf=-math.inf, neginf=-math.inf)
    list_rank = []
    list_com = []
    for j in range(len(index_test)-1, 0, -1):
        # n_comp = len(result_[index_test[j-1]:index_test[j]])
        COMP = COMPANY[index_test[j-1]:index_test[j]]
        rank_thuc = np.argsort(-result_[index_test[j-1]:index_test[j]]) + 1
        list_rank.append(rank_thuc[0])
        list_com.append(COMP[rank_thuc[0]-1])
    return list_rank, list_com

def get_in4_fomula(fomula):
    result_ =  np.nan_to_num(eval(fomula), nan=-math.inf, posinf=-math.inf, neginf=-math.inf)
    list_top_comp = []
    for j in range(len(index_test)-1, 0, -1):
        rank_thuc = np.argsort(-result_[index_test[j-1]:index_test[j]]) + 1
        list_top_comp.append(rank_thuc[:TOP_COMP_PER_QUARTER])
    return np.array(list_top_comp).flatten()

IN4_CT1_INDEX = 0
IN4_CT2_INDEX = ALL_QUARTER*TOP_COMP_PER_QUARTER
HISTORY_AGENT_INDEX= ALL_QUARTER*TOP_COMP_PER_QUARTER*2
CURRENT_QUARTER_INDEX = ALL_QUARTER*TOP_COMP_PER_QUARTER*2 + ALL_QUARTER
P_IN4_CT1 = 0
P_IN4_CT2 = 480

def reset(list_fomula):
    global LIST_RANK_CT1, LIST_RANK_CT2
    '''
    Hàm này trả ra 2 công thức và list top5 comp qua từng quý của công thức và các thông tin cần thiết khác
    '''
    current_quarter = 0
    history_agent = np.zeros(ALL_QUARTER)
    env_state = np.concatenate((get_in4_fomula(list_fomula[0]), get_in4_fomula(list_fomula[1]), history_agent, np.array([current_quarter])))
    return env_state

@nb.njit
def state_to_player(env_state):
    '''
    Hàm này trả ra lịch sử kết quả của 2 công thức trong các quý trước đó
    '''
    player_state = np.zeros(2*NUMBER_QUARTER_HISTORY*TOP_COMP_PER_QUARTER)
    if env_state[CURRENT_QUARTER_INDEX] != 0:
        history_ct1 = env_state[max(IN4_CT1_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-24)):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)]
        history_ct2 = env_state[max(IN4_CT2_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-24)+IN4_CT2_INDEX):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)+IN4_CT2_INDEX]
        player_state[P_IN4_CT1:P_IN4_CT1+len(history_ct1)] = history_ct1
        player_state[P_IN4_CT2:P_IN4_CT2+len(history_ct2)] = history_ct2
    return player_state

@nb.njit
def step(action, env_state):
    result_quarter = 0
    if action == 0:
        result_quarter = LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])]
        # print('check', action, LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])], 'quarter', int(env_state[CURRENT_QUARTER_INDEX]))
    elif action == 1:
        result_quarter = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)]
        # print('check', action, env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)], 'quarter', int(env_state[CURRENT_QUARTER_INDEX]))
    else:
        result_quarter = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]
        # print('check', action, env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)], 'quarter', int(env_state[CURRENT_QUARTER_INDEX]))
    if result_quarter == 0:
        print('toang',action)
        raise Exception('toang action')
    rank_3_action = np.array([LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]])
    rank_3_action = np.sort(rank_3_action)
    top_action = np.where(rank_3_action == result_quarter)[0][0] + 1
    # print('quarter', int(env_state[CURRENT_QUARTER_INDEX]),'check', 1/top_action, 'action', action, 'topaction',rank_3_action)
    env_state[int(HISTORY_AGENT_INDEX+env_state[CURRENT_QUARTER_INDEX])] = 1/top_action
    env_state[CURRENT_QUARTER_INDEX] += 1     
    return env_state

def action_player(env_state, agent_player, temp_file, per_file):
    player_state = state_to_player(env_state)
    played_move,temp_file,per_file = agent_player(player_state, temp_file, per_file)
    return played_move,temp_file, per_file

def one_game(agent_player, list_fomula, temp_file, per_file):
    # print(list_fomula)
    env_state = reset(list_fomula)
    count_turn = 0
    while count_turn < ALL_QUARTER:
        action, temp_file, per_file = action_player(env_state,agent_player,temp_file, per_file)
        env_state = step(action, env_state)
        count_turn += 1
    all_result = env_state[HISTORY_AGENT_INDEX:HISTORY_AGENT_INDEX+ALL_QUARTER]
    result = gmean(all_result)
    return result, per_file

def normal_main(agent_player, all_fomula, times, per_file):
    global data_full
    count = np.zeros(times)
    all_id_fomula = np.arange(len(all_fomula))
    for van in range(times):
        temp_file = [[0],[0],[0],[0]]
        shuffle = np.random.choice(all_id_fomula, 2, replace=False)
        list_fomula = all_fomula[shuffle]
        result, file_per = one_game(agent_player, list_fomula, temp_file, per_file)
        count[van] = result
    return count, file_per

def player_random(player_state, temp_file, per_file):
    list_action = np.array([0,1,2])
    action = int(np.random.choice(list_action))
    return action, temp_file, per_file
