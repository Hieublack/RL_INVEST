from glob import glob
from tabnanny import check
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
index_test = get_index_T(data_full_load)
data_full = data_full_load.copy()
for id in index_test[:-1]:
    s = data_full_load.iloc[id]
    s[1] = 1
    s[2] = 'NOT_INVEST'
    # s[4:] = np.full(len(data_full.columns) - 4, 1)
    s[4:] = np.average(data_full.loc[id: , data_full.columns[4:]], axis=0)
    data_full = data_full.append(s)
data_full = data_full.sort_values(by=['TIME', 'PROFIT'], ascending=[False, False], ignore_index=True)
index_test = get_index_T(data_full)

all_fomula = np.array(pd.read_csv('congthuc.csv')['fomula'])
TOP_COMP_PER_QUARTER = 20
NUMBER_QUARTER_HISTORY = 24
ALL_QUARTER = len(np.unique(data_full['TIME']))

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
        # n_comp = len(profit_q)
        # if np.max(profit_q) <= 1:
        #     list_rank_ko_dau_tu.append(1)
        # elif np.min(profit_q) >= 1:
        #     list_rank_ko_dau_tu.append(n_comp + 1)
        # else:
        list_rank_ko_dau_tu.append(np.where(profit_q == 1)[0][0]+1)
    return np.array(list_rank_ko_dau_tu)
LIST_RANK_NOT_INVEST = get_rank_not_invest()
LIST_RANK_NOT_INVEST_TEMP = np.zeros(ALL_QUARTER)
LIST_RANK_CT1 = np.zeros(ALL_QUARTER)
LIST_RANK_NOT_INVEST_CT1 = np.zeros(ALL_QUARTER)
LIST_RANK_CT2 = np.zeros(ALL_QUARTER)
LIST_RANK_NOT_INVEST_CT2 = np.zeros(ALL_QUARTER)
LIST_PROFIT_CT1 = np.zeros(ALL_QUARTER)
LIST_PROFIT_CT2 = np.zeros(ALL_QUARTER)


def get_in4_rank_fomula(fomula):
    result_ =  np.nan_to_num(eval(fomula), nan=-math.inf, posinf=-math.inf, neginf=-math.inf)
    list_rank = []
    list_com = []
    for j in range(len(index_test)-1, 0, -1):
        COMP = COMPANY[index_test[j-1]:index_test[j]]
        rank_thuc = np.argsort(-result_[index_test[j-1]:index_test[j]]) + 1
        list_rank.append(rank_thuc[0])
        list_com.append(COMP[rank_thuc[0]-1])
    return list_rank, list_com

#Hàm này tính cả list_profit
# def get_in4_fomula(fomula):
#     result_ =  np.nan_to_num(eval(fomula), nan=-math.inf, posinf=-math.inf, neginf=-math.inf)
#     list_top_comp = []
#     list_profit = []
#     for j in range(len(index_test)-1, 0, -1):
#         rank_thuc = np.argsort(-result_[index_test[j-1]:index_test[j]]) + 1
#         list_profit.append(PROFIT[index_test[j-1]:index_test[j]][rank_thuc[0]-1])
#         list_top_comp.append(rank_thuc[:TOP_COMP_PER_QUARTER])
#     return np.array(list_top_comp).flatten(), np.array(list_profit)

def get_in4_fomula(fomula):
    global LIST_RANK_NOT_INVEST_TEMP
    result_ =  np.nan_to_num(eval(fomula), nan=-math.inf, posinf=-math.inf, neginf=-math.inf)
    list_top_comp = []
    list_rank_not_invest_ct = []
    for j in range(len(index_test)-1, 0, -1):
        top2 = heapq.nlargest(2,result_[index_test[j-1]:index_test[j]])         #lấy top 2 giá trị lớn nhất
        if top2[0] == top2[1] or np.max(result_[index_test[j-1]:index_test[j]]) == np.min(result_[index_test[j-1]:index_test[j]]):
            return np.zeros(1), 0
        rank_thuc = np.argsort(-result_[index_test[j-1]:index_test[j]]) + 1
        COMP = COMPANY[index_test[j-1]:index_test[j]]
        id_not_invest = np.where(COMP== 'NOT_INVEST')[0][0]+1
        
        list_rank_not_invest_ct.append(np.where(rank_thuc == id_not_invest)[0][0]+1)

        list_top_comp.append(rank_thuc[:TOP_COMP_PER_QUARTER])
    LIST_RANK_NOT_INVEST_TEMP = np.array(list_rank_not_invest_ct)
    return np.array(list_top_comp).flatten(), 1

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

P_IN4_CT1 = 0
P_IN4_CT2 = P_IN4_CT1 + TOP_COMP_PER_QUARTER*NUMBER_QUARTER_HISTORY
P_GMEAN_P1 = P_IN4_CT2 + TOP_COMP_PER_QUARTER*NUMBER_QUARTER_HISTORY
P_GMEAN_P2 = P_GMEAN_P1 + 1
P_ID_NOT_INVEST_CT1 = P_GMEAN_P2 + 1
P_ID_NOT_INVEST_CT2 = P_ID_NOT_INVEST_CT1 + 1


def reset():
    global LIST_RANK_CT1, LIST_RANK_CT2, LIST_PROFIT_CT1, LIST_PROFIT_CT2, LIST_RANK_NOT_INVEST_CT1, LIST_RANK_NOT_INVEST_CT2
    '''
    Hàm này trả ra 2 công thức và list top20 comp qua từng quý của công thức và các thông tin cần thiết khác
    '''
    # list_fomula = []
    count_fomula = 0
    while count_fomula < 2:
        fomula = create_fomula()
        temp, check = get_in4_fomula(fomula)
        # print('check getin4', len(temp), check)
        count_fomula += check
        if count_fomula == 1 and check == 1:
            LIST_RANK_CT1 = temp.copy()
            LIST_RANK_NOT_INVEST_CT1 = LIST_RANK_NOT_INVEST_TEMP.copy()
            # list_fomula.append(fomula)
        elif count_fomula == 2 and check == 1:
            LIST_RANK_CT2 = temp.copy()
            LIST_RANK_NOT_INVEST_CT2 = LIST_RANK_NOT_INVEST_TEMP.copy()
            # list_fomula.append(fomula)
    id_not_invest_ct1 = LIST_RANK_NOT_INVEST_CT1[0]
    id_not_invest_ct2 = LIST_RANK_NOT_INVEST_CT2[0]
    current_quarter = 0
    id_action = 0
    check_end_game = 0
    history_agent = np.zeros(ALL_QUARTER*3)
    # LIST_RANK_CT1, LIST_PROFIT_CT1 = get_in4_fomula(list_fomula[0])
    # LIST_RANK_CT2, LIST_PROFIT_CT2 = get_in4_fomula(list_fomula[1])
    env_state = np.concatenate((LIST_RANK_CT1, LIST_RANK_CT2, history_agent, np.array([id_not_invest_ct1, id_not_invest_ct2, current_quarter, id_action, check_end_game])))
    return env_state

@nb.njit
def state_to_player(env_state):
    '''
    Hàm này trả ra lịch sử kết quả của 2 công thức trong các quý trước đó
    '''
    id_action = env_state[ID_ACTION_INDEX]
    player_state = np.zeros(2*NUMBER_QUARTER_HISTORY*TOP_COMP_PER_QUARTER + 4)
    player_state[P_ID_NOT_INVEST_CT1] = env_state[ID_NOT_INVEST_CT1]
    player_state[P_ID_NOT_INVEST_CT2] = env_state[ID_NOT_INVEST_CT2]
    if env_state[CURRENT_QUARTER_INDEX] != 0:
        history_ct1 = env_state[max(IN4_CT1_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-24)):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)]
        history_ct2 = env_state[max(IN4_CT2_INDEX, TOP_COMP_PER_QUARTER*(env_state[CURRENT_QUARTER_INDEX]-24)+IN4_CT2_INDEX):int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)+IN4_CT2_INDEX]
        player_state[P_IN4_CT1:P_IN4_CT1+len(history_ct1)] = history_ct1
        player_state[P_IN4_CT2:P_IN4_CT2+len(history_ct2)] = history_ct2
    agent_history = env_state[HISTORY_AGENT_INDEX : HISTORY_AGENT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX : HISTORY_SYS_BOT_INDEX+ALL_QUARTER][:int(env_state[CURRENT_QUARTER_INDEX])]
    if env_state[CHECK_END_INDEX] == 1:
        if id_action == 0:
            player_state[P_GMEAN_P1] = np.exp(np.mean(np.log(agent_history)))
            player_state[P_GMEAN_P2] = np.exp(np.mean(np.log(sys_bot_history)))
        else:
            player_state[P_GMEAN_P1] = np.exp(np.mean(np.log(sys_bot_history)))
            player_state[P_GMEAN_P2] = np.exp(np.mean(np.log(agent_history)))
    return player_state

@nb.njit
def step(action, env_state):
    global LIST_RANK_CT1, LIST_RANK_CT2, LIST_PROFIT_CT1, LIST_PROFIT_CT2, LIST_RANK_NOT_INVEST_CT1, LIST_RANK_NOT_INVEST_CT2
    id_action = env_state[ID_ACTION_INDEX]
    result_quarter = 0
    if action == 0:
        result_quarter = LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])]
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
    rank_3_action = np.array([LIST_RANK_NOT_INVEST[int(env_state[CURRENT_QUARTER_INDEX])], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER)], env_state[int(env_state[CURRENT_QUARTER_INDEX]*TOP_COMP_PER_QUARTER+IN4_CT2_INDEX)]])
    rank_3_action = np.sort(rank_3_action)
    top_action = np.where(rank_3_action == result_quarter)[0][0] + 1
    # print('quarter', int(env_state[CURRENT_QUARTER_INDEX]),'check', 1/top_action, 'action', action, 'topaction',rank_3_action)
    env_state[int(HISTORY_AGENT_INDEX + ALL_QUARTER*id_action +env_state[CURRENT_QUARTER_INDEX])] = (4-top_action)/3
    if env_state[ID_ACTION_INDEX] == 1:
        env_state[CURRENT_QUARTER_INDEX] += 1  
        #rank giá trị công thức của việc không đầu tư
        if env_state[CURRENT_QUARTER_INDEX] < ALL_QUARTER:
            env_state[ID_NOT_INVEST_CT1] = LIST_RANK_NOT_INVEST_CT1[int(env_state[CURRENT_QUARTER_INDEX])]
            env_state[ID_NOT_INVEST_CT2] = LIST_RANK_NOT_INVEST_CT2[int(env_state[CURRENT_QUARTER_INDEX])]
        env_state[ID_ACTION_INDEX] = 0
    else:
        env_state[ID_ACTION_INDEX] = 1

    return env_state

def action_player(env_state, list_player, temp_file, per_file):
    player_state = state_to_player(env_state)
    current_player = int(env_state[ID_ACTION_INDEX])
    played_move,temp_file,per_file = list_player[current_player](player_state, temp_file, per_file)
    return played_move,temp_file, per_file
    
@nb.njit
def check_winner(env_state):
    agent_history = env_state[HISTORY_AGENT_INDEX : HISTORY_AGENT_INDEX+ALL_QUARTER]
    sys_bot_history = env_state[HISTORY_SYS_BOT_INDEX : HISTORY_SYS_BOT_INDEX+ALL_QUARTER]
    np.exp(np.mean(np.log(sys_bot_history)))
    if np.exp(np.mean(np.log(agent_history))) > np.exp(np.mean(np.log(sys_bot_history))): return 0
    else: return 1

@nb.njit
def check_victory(player_state):
    if not (player_state[P_GMEAN_P1] == player_state[P_GMEAN_P2] and player_state[P_GMEAN_P2] == 0):
        if player_state[P_GMEAN_P1] > player_state[P_GMEAN_P2]: return 1
        else: return 0
    else: return -1

def create_fomula():
    all_char = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    all_char = 'BCDEFGHIJKLMNOPQRSTUVWXYZ'
    power = random.randint(1, 10)
    # print('bậc: ', power)
    operand = random.randint(1, 10)
    # print('số toán hạng: ', operand)
    list_exp_child = []
    ct = '('
    for i in range(operand):
        ct += random.choice('+-')
        numerator = random.randint(power, 25-power)
        denominator = numerator - power
        numer_char = random.choices(all_char, k = numerator)
        ct += '(' + '*'.join(numer_char) + ')'
        # print(ct, 'mẫu', denominator)
        if denominator > 0:
            all_denom_char = list(set(numer_char)^set(all_char))
            # print('CHECK: ', len(all_denom_char) , denominator)
            if len(all_denom_char) < denominator:
                all_denom_char += random.choices(all_char, k = (denominator - len(all_denom_char)))
            denom_char = random.choices(all_denom_char, k = denominator)
            ct += '/(' + '*'.join(numer_char) + ')'
    ct += ')' + '/A'*power
    return ct

def one_game(list_player, temp_file, per_file):
    env_state = reset()
    count_turn = 0
    while count_turn < ALL_QUARTER*2:
        action, temp_file, per_file = action_player(env_state, list_player, temp_file, per_file)
        env_state = step(action, env_state)
        count_turn += 1
    env_state[CHECK_END_INDEX] = 1
    for id_player in range(len(list_player)):
        action, temp_file, per_file = action_player(env_state,list_player,temp_file, per_file)
        env_state[ID_ACTION_INDEX] = (env_state[ID_ACTION_INDEX] + 1)%len(list_player)
    result = check_winner(env_state)
    return result, per_file

def normal_main(agent_player, times, per_file):
    global data_full
    count = np.zeros(2)
    # all_id_fomula = np.arange(len(all_fomula))
    list_player = [agent_player, player_random]
    for van in range(times):
        temp_file = [[0],[0]]
        # shuffle = np.random.choice(all_id_fomula, 2, replace=False)
        # list_fomula = all_fomula[shuffle]
        winner, file_per = one_game(list_player, temp_file, per_file)
        if winner == 0:
            count[0] += 1
        else:
            count[1] += 1
    return count, file_per

def player_random1(player_state, temp_file, per_file):
    list_action = np.array([0,1,2])
    action = int(np.random.choice(list_action))
    # print('check: ', player_state[P_ID_NOT_INVEST_CT1], player_state[P_ID_NOT_INVEST_CT2])
    check = check_victory(player_state)
    return action, temp_file, per_file

def player_random(player_state, temp_file, per_file):
    list_action = np.array([0,1,2])
    action = int(np.random.choice(list_action))
    # check = check_victory(player_state)
    return action, temp_file, per_file
