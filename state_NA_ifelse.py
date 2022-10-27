import pandas as pd
import math
import numpy as np
import heapq
from scipy.stats.mstats import gmean, hmean
import numba as nb
import random

data_full_load = pd.read_csv('data_test.csv')
data_full_load = data_full_load.sort_values(by=['TIME', 'PROFIT'], ascending=[False, False], ignore_index=True)
def get_index_T(data_full):
    list_T = data_full['TIME']
    index_T = [0]
    for i in range(len(list_T)-1):
        if list_T[i] != list_T[i+1]:
            index_T.append(i+1)
    index_T.append(len(list_T))
    return index_T 

def get_in4_rank_fomula(result_fomula):
    list_rank = []
    list_com = []
    for j in range(len(index_test)-1, 0, -1):
        COMP = COMPANY[index_test[j-1]:index_test[j]]
        rank_thuc = np.argsort(-result_fomula[index_test[j-1]:index_test[j]]) + 1
        list_rank.append(rank_thuc[0])
        list_com.append(COMP[rank_thuc[0]-1])
    return list_rank, list_com

index_test = get_index_T(data_full_load)
data_full = data_full_load.copy()
for id in index_test[:-1]:
    s = data_full_load.iloc[id]
    s[1] = 1
    s[2] = 'NOT_INVEST'
    # s[4:] = np.full(len(data_full.columns) - 4, 1)
    s[4:] = np.average(data_full.loc[id: , data_full.columns[4:]], axis=0)
    data_full = data_full.append(s)

data_full = data_full.sort_values(by=['TIME', 'PROFIT'], ascending=[False, True], ignore_index=True)
index_test = np.array(get_index_T(data_full))
PROFIT = np.array(data_full["PROFIT"])
COMPANY = np.array(data_full["SYMBOL"])
data_arr = np.array(data_full[data_full.columns[4:]]).T

NUMBER_VARIABLE = len(data_arr)
TOP_COMP_PER_QUARTER = 20
NUMBER_QUARTER_HISTORY = 1
START_QUARTER = 1
ALL_QUARTER = len(np.unique(data_full['TIME']))



def get_rank_not_invest():
    list_rank_ko_dau_tu = []
    for j in range(len(index_test)-1, 0, -1):
        # profit_q = PROFIT[index_test[j-1]:index_test[j]]
        COMP = COMPANY[index_test[j-1]:index_test[j]]
        list_rank_ko_dau_tu.append(np.where(COMP == 'NOT_INVEST')[0][0]+1)
    return np.array(list_rank_ko_dau_tu)

LIST_RANK_NOT_INVEST = get_rank_not_invest()

LIST_PROFIT_CT1 = np.zeros(ALL_QUARTER)
LIST_PROFIT_CT2 = np.zeros(ALL_QUARTER)

def save_data():
    np.save('index_test.npy', index_test)
    np.save('data.npy', data_arr)
    np.save('list_rank_profit_not_invest.npy', LIST_RANK_NOT_INVEST)
    np.save('list_company.npy', COMPANY)

save_data()

@nb.njit()
def get_in4_fomula(result_fomula, list_rank_not_invest_temp, ALL_IN4_SYS):
    list_top_comp = np.array([-1])
    list_rank_not_invest_ct = np.array([-1])
    
    for j in range(len(index_test)-1, 0, -1):
        top2 = heapq.nlargest(2,result_fomula[index_test[j-1]:index_test[j]])         #lấy top 2 giá trị lớn nhất
        if top2[0] == top2[1] or np.max(result_fomula[index_test[j-1]:index_test[j]]) == np.min(result_fomula[index_test[j-1]:index_test[j]]):
            # print('toang cong thuc', top2, np.max(result_fomula[index_test[j-1]:index_test[j]]), np.min(result_fomula[index_test[j-1]:index_test[j]]))
            return np.array([-1]), np.array([-1]), 0
        rank_thuc = np.argsort(-result_fomula[index_test[j-1]:index_test[j]]) + 1
        id_not_invest = ALL_IN4_SYS[0][-j] 
        # print(ALL_IN4_SYS[0][-j], LIST_RANK_NOT_INVEST[-j] )
        if list_rank_not_invest_ct[0] == -1:
            list_rank_not_invest_ct = np.array([np.where(rank_thuc == id_not_invest)[0][0]+1])
            list_top_comp = rank_thuc[:TOP_COMP_PER_QUARTER]
        else:
            list_rank_not_invest_ct = np.append(list_rank_not_invest_ct, np.where(rank_thuc == id_not_invest)[0][0]+1)
            list_top_comp = np.append(list_top_comp, rank_thuc[:TOP_COMP_PER_QUARTER])
    list_rank_not_invest_temp = list_rank_not_invest_ct
    return list_top_comp, list_rank_not_invest_temp, 1

IN4_CT1_INDEX = 0
IN4_CT2_INDEX = ALL_QUARTER*TOP_COMP_PER_QUARTER
HISTORY_AGENT_INDEX= IN4_CT2_INDEX + ALL_QUARTER*TOP_COMP_PER_QUARTER
HISTORY_SYS_BOT_INDEX = HISTORY_AGENT_INDEX + ALL_QUARTER
HISTORY_PROFIT_AGENT = HISTORY_SYS_BOT_INDEX + ALL_QUARTER
ID_NOT_INVEST_CT1 = HISTORY_PROFIT_AGENT + ALL_QUARTER
ID_NOT_INVEST_CT2 = ID_NOT_INVEST_CT1 + 1
CURRENT_QUARTER_INDEX = ID_NOT_INVEST_CT2 + 1
ID_ACTION_INDEX = CURRENT_QUARTER_INDEX + 1
CHECK_END_INDEX = ID_ACTION_INDEX + 1
NUMBER_COMP_LAST_INDEX = CHECK_END_INDEX + 1
NUMBER_COMP_CURENT_INDEX = NUMBER_COMP_LAST_INDEX + 1
RANK_NOT_INVEST = NUMBER_COMP_CURENT_INDEX + 1
RATIO_CT1_NOT_INVEST = RANK_NOT_INVEST + 1
RATIO_CT2_NOT_INVEST = RATIO_CT1_NOT_INVEST + 1

P_IN4_CT1 = 0
P_IN4_CT2 = P_IN4_CT1 + TOP_COMP_PER_QUARTER*NUMBER_QUARTER_HISTORY
P_GMEAN_P1 = P_IN4_CT2 + TOP_COMP_PER_QUARTER*NUMBER_QUARTER_HISTORY
P_GMEAN_P2 = P_GMEAN_P1 + 1
P_RANK_NOT_INVEST = P_GMEAN_P2 + 1
P_RATIO_CT1_NOT_INVEST = P_RANK_NOT_INVEST + 1 
P_RATIO_CT2_NOT_INVEST = P_RATIO_CT1_NOT_INVEST + 1
P_ID_NOT_INVEST_CT1 = P_RATIO_CT2_NOT_INVEST + 1
P_ID_NOT_INVEST_CT2 = P_ID_NOT_INVEST_CT1 + 1
P_NUMBER_COMP_LAST_INDEX = P_ID_NOT_INVEST_CT2+ 1
P_NUMBER_COMP_CURRENT_INDEX = P_NUMBER_COMP_LAST_INDEX + 1

# @nb.njit()
def reset(ALL_IN4_SYS, LIST_ALL_COMP_PER_QUARTER):
    # global LIST_RANK_CT1, LIST_RANK_CT2, LIST_PROFIT_CT2, LIST_RANK_NOT_INVEST_CT1, LIST_RANK_NOT_INVEST_CT2
    '''
    Hàm này trả ra 2 công thức và list top20 comp qua từng quý của công thức và các thông tin cần thiết khác
    '''
    LIST_RANK_CT1 = np.zeros(ALL_QUARTER)
    LIST_RANK_CT2 = np.zeros(ALL_QUARTER)
    # list_fomula = []
    count_fomula = 0
    while count_fomula < 2:
        result_fomula = create_fomula(data_arr)
        LIST_RANK_NOT_INVEST_TEMP = np.zeros(ALL_QUARTER)
        temp, LIST_RANK_NOT_INVEST_TEMP, check = get_in4_fomula(result_fomula, LIST_RANK_NOT_INVEST_TEMP, ALL_IN4_SYS)
        count_fomula += check
        if count_fomula == 1 and check == 1:
            LIST_RANK_CT1 = temp.copy()
            ALL_IN4_SYS[1] = LIST_RANK_NOT_INVEST_TEMP.copy() 
            # list_fomula.append(fomula)
        elif count_fomula == 2 and check == 1:
            LIST_RANK_CT2 = temp.copy()
            ALL_IN4_SYS[2] = LIST_RANK_NOT_INVEST_TEMP.copy() 
            # list_fomula.append(fomula)

    history_agent = np.zeros(ALL_QUARTER*3)
    id_not_invest_ct1 = ALL_IN4_SYS[1][0]
    id_not_invest_ct2 = ALL_IN4_SYS[2][0]
    current_quarter = 1
    id_action = 0
    check_end_game = 0
    number_comp_last = LIST_ALL_COMP_PER_QUARTER[current_quarter-1]
    number_comp_current = LIST_ALL_COMP_PER_QUARTER[current_quarter]
    rank_not_invest = ALL_IN4_SYS[0][0]
    ratio_ct1_not_invest = LIST_RANK_CT1[0]/rank_not_invest
    ratio_ct2_not_invest = LIST_RANK_CT2[0]/rank_not_invest

    env_state = np.concatenate((LIST_RANK_CT1, LIST_RANK_CT2, history_agent,
                         np.array([id_not_invest_ct1, id_not_invest_ct2, 
                         current_quarter, id_action, check_end_game,
                          number_comp_last,number_comp_current , 
                          rank_not_invest, ratio_ct1_not_invest, 
                          ratio_ct2_not_invest])))
    return env_state, ALL_IN4_SYS

@nb.njit()
def create_fomula(data_arr):
    power = np.random.randint(1, 10)
    operand = np.random.randint(1, 10)
    result_fomula = np.zeros(data_arr.shape[1])
    # ct = []
    for i in range(operand):
        op = np.random.randint(2)
        # ct.append(op)
        numerator = np.random.randint(power, NUMBER_VARIABLE - 1 - power)
        denominator = numerator - power
        numer_var = np.random.randint(1, NUMBER_VARIABLE, numerator)
        result_temp = np.zeros(data_arr.shape[1])+1
        if denominator > 0:
            all_var = np.arange(1,NUMBER_VARIABLE)
            for id in range(len(all_var)):
                if all_var[id] in numer_var:
                    all_var[id] = 0
            all_denom_var = all_var[all_var > 0]
            if len(all_denom_var) < denominator:
                all_denom_var = np.append(all_denom_var, np.random.choice(all_var, denominator - len(all_denom_var)))
            denom_var = np.random.choice(all_denom_var, denominator)
            denom_var = np.append(denom_var, np.zeros(numerator-denominator).astype(np.int64))
            denom_var = denom_var.astype(np.int64)
            # ct.append([list(numer_var), list(denom_var)])
            for idx in range(len(numer_var)):
                num = data_arr[numer_var[idx]]
                denom = data_arr[denom_var[idx]]
                denom_zero = np.where(denom == 0)[0]
                denom[denom_zero] = 1
                num[denom_zero] = 1
                result_temp =  result_temp*(num/denom)
        else:
            denom_var = np.zeros(numerator).astype(np.int64)
            # ct.append([list(numer_var), list(denom_var)])
            for id in range(len(numer_var)):
                num = data_arr[numer_var[id]]
                denom = data_arr[denom_var[id]]
                denom_zero = np.where(denom == 0)[0]
                denom[denom_zero] = 1
                num[denom_zero] = 1
                result_temp =  result_temp*(num/denom)
        if op == 1:
            result_fomula = result_fomula + result_temp
        else:
            result_fomula = result_fomula - result_temp
    return result_fomula

@nb.njit()
def state_to_player(env_state):
    '''
    Hàm này trả ra lịch sử kết quả của 2 công thức trong các quý trước đó
    '''
    id_action = env_state[ID_ACTION_INDEX]
    player_state = np.zeros(2*NUMBER_QUARTER_HISTORY*TOP_COMP_PER_QUARTER + 9)
    player_state[P_ID_NOT_INVEST_CT1] = env_state[ID_NOT_INVEST_CT1]
    player_state[P_ID_NOT_INVEST_CT2] = env_state[ID_NOT_INVEST_CT2]
    if env_state[CURRENT_QUARTER_INDEX] != 0:
        history_ct1 = env_state[max(IN4_CT1_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-NUMBER_QUARTER_HISTORY)):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)]
        history_ct2 = env_state[max(IN4_CT2_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-NUMBER_QUARTER_HISTORY)+IN4_CT2_INDEX):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)+IN4_CT2_INDEX]
        len_bonus = int(TOP_COMP_PER_QUARTER * (NUMBER_QUARTER_HISTORY - env_state[CURRENT_QUARTER_INDEX]))
        if len_bonus > 0:
            a = np.zeros(len_bonus)
            history_ct1 = np.append(a, history_ct1)
            history_ct2 = np.append(a, history_ct2)
        player_state[P_IN4_CT1:P_IN4_CT2] = history_ct1
        player_state[P_IN4_CT2:P_GMEAN_P1] = history_ct2
    # agent_history = env_state[HISTORY_AGENT_INDEX + STAER_QUARTER: HISTORY_AGENT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    # sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX + STAER_QUARTER: HISTORY_SYS_BOT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    if env_state[CHECK_END_INDEX] == 1:
        agent_history = env_state[HISTORY_AGENT_INDEX + START_QUARTER: HISTORY_AGENT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
        sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX + START_QUARTER: HISTORY_SYS_BOT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
        if id_action == 0:
            # player_state[P_GMEAN_P1] = np.exp(np.mean(np.log(agent_history)))
            # player_state[P_GMEAN_P2] = np.exp(np.mean(np.log(sys_bot_history)))
            player_state[P_GMEAN_P1] = len(agent_history)/np.sum(1/agent_history)
            player_state[P_GMEAN_P2] = len(sys_bot_history)/np.sum(1/sys_bot_history)
        else:
            # player_state[P_GMEAN_P1] = np.exp(np.mean(np.log(sys_bot_history)))
            # player_state[P_GMEAN_P2] = np.exp(np.mean(np.log(agent_history)))
            player_state[P_GMEAN_P1] = len(sys_bot_history)/np.sum(1/sys_bot_history)
            player_state[P_GMEAN_P2] = len(agent_history)/np.sum(1/agent_history)
        # print(player_state[P_GMEAN_P1], player_state[P_GMEAN_P2])
    player_state[P_RANK_NOT_INVEST] = env_state[RANK_NOT_INVEST]
    player_state[P_RATIO_CT1_NOT_INVEST] = env_state[RATIO_CT1_NOT_INVEST]
    player_state[P_RATIO_CT2_NOT_INVEST] = env_state[RATIO_CT2_NOT_INVEST]
    player_state[P_NUMBER_COMP_LAST_INDEX] = env_state[NUMBER_COMP_LAST_INDEX]
    player_state[P_NUMBER_COMP_CURRENT_INDEX] = env_state[NUMBER_COMP_CURENT_INDEX]
    return player_state

@nb.njit()
def step(action, env_state, ALL_IN4_SYS, LIST_ALL_COMP_PER_QUARTER):
    # print('action step: ', action, env_state[CURRENT_QUARTER_INDEX])
    id_action = env_state[ID_ACTION_INDEX]
    result_quarter = 0
    if action == 0:
        result_quarter = ALL_IN4_SYS[0][int(env_state[CURRENT_QUARTER_INDEX])]
        # env_state[int(HISTORY_PROFIT_AGENT+env_state[CURRENT_QUARTER_INDEX])] = 1
    elif action == 1:
        result_quarter = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)]
        # env_state[int(HISTORY_PROFIT_AGENT+env_state[CURRENT_QUARTER_INDEX])] = LIST_PROFIT_CT1[int(env_state[CURRENT_QUARTER_INDEX])]
    else:
        result_quarter = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]
        # env_state[int(HISTORY_PROFIT_AGENT+env_state[CURRENT_QUARTER_INDEX])] = LIST_PROFIT_CT2[int(env_state[CURRENT_QUARTER_INDEX])]
    if result_quarter == 0:
        # print('toang',action,  env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)], np.min(LIST_RANK_CT2))
        raise Exception('toang action')
    # print(action, result_quarter, env_state[NUMBER_COMP_INDEX])
    '''
    nếu xét action theo hạng của action
    rank_3_action = np.array([LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]])
    rank_3_action = np.sort(rank_3_action)
    top_action = np.where(rank_3_action == result_quarter)[0][0] + 1
    print('quarter', int(env_state[CURRENT_QUARTER_INDEX]),'check', 1/top_action, 'action', action, 'topaction',rank_3_action)
    env_state[int(HISTORY_AGENT_INDEX + ALL_QUARTER*id_action +env_state[CURRENT_QUARTER_INDEX])] = (4-top_action)/3
    '''
    #nếu xét: rank theo action/rank của action 0 (rank action 0 là rank theo lợi nhuận của việc không đầu tư)
    result_action = result_quarter/ALL_IN4_SYS[0][int(env_state[CURRENT_QUARTER_INDEX])]
    env_state[int(HISTORY_AGENT_INDEX + ALL_QUARTER*id_action +env_state[CURRENT_QUARTER_INDEX])] = result_action
    env_state[RATIO_CT1_NOT_INVEST] = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)] / ALL_IN4_SYS[0][int(env_state[CURRENT_QUARTER_INDEX])]
    env_state[RATIO_CT2_NOT_INVEST] = env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]/ALL_IN4_SYS[0][int(env_state[CURRENT_QUARTER_INDEX])]
    if env_state[ID_ACTION_INDEX] == 1:
        env_state[CURRENT_QUARTER_INDEX] += 1  
        #rank giá trị công thức của việc không đầu tư
        if env_state[CURRENT_QUARTER_INDEX] < ALL_QUARTER:
            env_state[RANK_NOT_INVEST] = ALL_IN4_SYS[0][int(env_state[CURRENT_QUARTER_INDEX] - 1)]
            env_state[ID_NOT_INVEST_CT1] = ALL_IN4_SYS[1][int(env_state[CURRENT_QUARTER_INDEX])]
            env_state[ID_NOT_INVEST_CT2] = ALL_IN4_SYS[2][int(env_state[CURRENT_QUARTER_INDEX])]
        env_state[NUMBER_COMP_CURENT_INDEX] = LIST_ALL_COMP_PER_QUARTER[int(env_state[CURRENT_QUARTER_INDEX])]
        env_state[NUMBER_COMP_LAST_INDEX] = LIST_ALL_COMP_PER_QUARTER[int(env_state[CURRENT_QUARTER_INDEX])-1]
        env_state[ID_ACTION_INDEX] = 0
    else:
        env_state[ID_ACTION_INDEX] = 1
    # print('step', env_state[-10:], env_state[ID_ACTION_INDEX], env_state[CURRENT_QUARTER_INDEX])

    return env_state

# @nb.njit()
def action_player(env_state, list_player, temp_file, per_file):
    player_state = state_to_player(env_state)
    current_player = int(env_state[ID_ACTION_INDEX])
    played_move,temp_file[current_player],per_file = list_player[current_player](player_state, temp_file[current_player], per_file)
    if played_move not in [0, 1, 2]:
        raise Exception('Action false')
    return played_move,temp_file, per_file
    
@nb.njit()
def check_winner(env_state):
    # agent_history = env_state[HISTORY_AGENT_INDEX : HISTORY_AGENT_INDEX+ALL_QUARTER]
    # sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX : HISTORY_SYS_BOT_INDEX+ALL_QUARTER]
    agent_history = env_state[HISTORY_AGENT_INDEX + START_QUARTER: HISTORY_AGENT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX + START_QUARTER: HISTORY_SYS_BOT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    agent_result = np.exp(np.mean(np.log(agent_history)))
    sys_result = np.exp(np.mean(np.log(sys_bot_history)))
    # agent_result = len(agent_history)/np.sum(1/agent_history)
    # sys_result =  len(sys_bot_history)/np.sum(1/sys_bot_history)
    # print(agent_result, sys_result)
    if agent_result > sys_result: return 0
    else: return 1

@nb.njit()
def check_victory(player_state):
    if not (player_state[P_GMEAN_P1] == player_state[P_GMEAN_P2] and player_state[P_GMEAN_P2] == 0):
        if player_state[P_GMEAN_P1] > player_state[P_GMEAN_P2]: return 1
        else: return 0
    else: return -1

def one_game(list_player, temp_file, per_file, LIST_RANK_NOT_INVEST, LIST_ALL_COMP_PER_QUARTER, list_all_result):
    ALL_IN4_SYS = np.array([LIST_RANK_NOT_INVEST, np.zeros(ALL_QUARTER), np.zeros(ALL_QUARTER)])
    env_state, ALL_IN4_SYS = reset(ALL_IN4_SYS, LIST_ALL_COMP_PER_QUARTER)
    # print('one',env_state[-10:], env_state[ID_ACTION_INDEX], env_state[CURRENT_QUARTER_INDEX])
    count_turn = 0
    while env_state[CURRENT_QUARTER_INDEX] < ALL_QUARTER:
        action, temp_file, per_file = action_player(env_state, list_player, temp_file, per_file)
        env_state = step(action, env_state, ALL_IN4_SYS, LIST_ALL_COMP_PER_QUARTER)
        count_turn += 1
    env_state[CHECK_END_INDEX] = 1
    for id_player in range(len(list_player)):
        action, temp_file, per_file = action_player(env_state,list_player,temp_file, per_file)
        env_state[ID_ACTION_INDEX] = (env_state[ID_ACTION_INDEX] + 1)%len(list_player)
    result = check_winner(env_state)
    agent_history = env_state[HISTORY_AGENT_INDEX + START_QUARTER: HISTORY_AGENT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX + START_QUARTER: HISTORY_SYS_BOT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    # print(START_QUARTER)
    # agent_result = len(agent_history)/np.sum(1/agent_history)
    # sys_result =  len(sys_bot_history)/np.sum(1/sys_bot_history)
    
    agent_result = np.exp(np.mean(np.log(agent_history)))
    sys_result = np.exp(np.mean(np.log(sys_bot_history)))
    list_all_result[0].append(agent_result)
    list_all_result[1].append(sys_result)

    # print(env_state[HISTORY_AGENT_INDEX:])
    return result, per_file

@nb.njit()
def amount_action():
    return  3

@nb.njit()
def get_list_action(state):
    return np.full(amount_action(), 1)

@nb.njit()
def amount_state():
    return 49


def normal_main(agent_player, times, per_file):
    global data_full, data_arr, LIST_ALL_COMP_PER_QUARTER
    count = np.zeros(2)
    # all_id_fomula = np.arange(len(all_fomula))
    list_player = [agent_player, player_random]
    LIST_RANK_NOT_INVEST = get_rank_not_invest()
    LIST_ALL_COMP_PER_QUARTER = []
    for j in range(len(index_test)-1, 0, -1):
        LIST_ALL_COMP_PER_QUARTER.append(index_test[j] - index_test[j-1])
    LIST_ALL_COMP_PER_QUARTER.append(LIST_ALL_COMP_PER_QUARTER[-1])
    LIST_ALL_COMP_PER_QUARTER = np.array(LIST_ALL_COMP_PER_QUARTER)

    list_all_result = [[], []]
    for van in range(times):
        temp_file = [[0],[0]]
        # shuffle = np.random.choice(all_id_fomula, 2, replace=False)
        # list_fomula = all_fomula[shuffle]
        winner, file_per = one_game(list_player, temp_file, per_file, LIST_RANK_NOT_INVEST, LIST_ALL_COMP_PER_QUARTER, list_all_result)
        if winner == 0:
            count[0] += 1
        else:
            count[1] += 1
    # print(list_all_result)
    print(f'Average agent and system: {np.average(list_all_result[0])} {np.average(list_all_result[1])}')
    return count, file_per

def player_random1(player_state, temp_file, per_file):
    list_action = np.array([0,1,2])
    action = int(np.random.choice(list_action))
    check = check_victory(player_state)
    if check == 1:
        print(player_state[-20:])
    return action, temp_file, per_file

def player_random(player_state, temp_file, per_file):
    list_action = np.array([0,1,2])
    action = int(np.random.choice(list_action))
    # check = check_victory(player_state)
    # print(player_state)
    return action, temp_file, per_file

