import random
import numpy as np
import setting as s
import time
import os
from collections import Counter

start = time.time()


# note: only learn primitive_y
# ----------------------------------------------------------------------------------------------------------------------
def func_update_primitive(world, primitive_succ_lib):
    # find the same attribute
    key = [world['width'], world['height'], world['density']]
    key = tuple(key)
    if key in primitive_succ_lib:
        values = primitive_succ_lib[key]
        np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
        value = random.choice(values)
        world['primitive_y'] = round(value, 2)
    else:
        np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
        world['primitive_y'] = np.random.random() * world['height']


# ----------------------------------------------------------------------------------------------------------------------
def func_select_optimal_tp(paras):
    # tp0: 5, 3, 0 (place 5 objects and then push it; place 3 objects and then push it)
    # tp1: 4, 4, 0
    # tp2: 4, 3, 1
    # tp3: 4, 2, 2
    # tp4: 3, 3, 2
    # tp5: 5, 2, 1
    # if task is tp0: 5, 3, 0, then its utility is:
    cost_tp0 = s.cost_place * 8. + s.cost_push * 2. + s.cost_pick * 0.
    score1_tp0 = (paras[0] + paras[1] + paras[2] + paras[3] + paras[4] + paras[0] + paras[1] + paras[2]) * s.score_place
    score2_tp0 = (paras[0] * paras[1] * paras[2] * paras[3] * paras[4] * paras[0] * paras[1] * paras[2]) * s.score_push
    score_tp0 = score1_tp0 + score2_tp0
    penalty_tp0 = (1 - paras[0] * paras[1] * paras[2] * paras[3] * paras[4] * paras[0] * paras[1] * paras[
        2]) * s.penalty_value
    utility_tp0 = score_tp0 - cost_tp0 - penalty_tp0

    # if task is tp1: 4, 4, 0, then its utility is:
    cost_tp1 = s.cost_place * 8. + s.cost_push * 2. + s.cost_pick * 0.
    score1_tp1 = (paras[0] + paras[1] + paras[2] + paras[3] + paras[0] + paras[1] + paras[2] + paras[3]) * s.score_place
    score2_tp1 = (paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[2] * paras[3]) * s.score_push
    score_tp1 = score1_tp1 + score2_tp1
    penalty_tp1 = (1 - paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[2] * paras[
        3]) * s.penalty_value
    utility_tp1 = score_tp1 - cost_tp1 - penalty_tp1

    # if task is tp2: 4, 3, 1, then its utility is:
    cost_tp2 = s.cost_place * 7. + s.cost_push * 2. + s.cost_pick * 1.
    score1_tp2 = (paras[0] + paras[1] + paras[2] + paras[3] + paras[0] + paras[1] + paras[2] + paras[0]) * s.score_place
    score2_tp2 = (paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[2] * paras[0]) * s.score_push
    score_tp2 = score1_tp2 + score2_tp2
    penalty_tp2 = (1 - paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[2] * paras[
        0]) * s.penalty_value
    utility_tp2 = score_tp2 - cost_tp2 - penalty_tp2

    # if task is tp3: 4, 2, 2, then its utility is:
    cost_tp3 = s.cost_place * 8. + s.cost_push * 3. + s.cost_pick * 0.
    score1_tp3 = (paras[0] + paras[1] + paras[2] + paras[3] + paras[0] + paras[1] + paras[0] + paras[1]) * s.score_place
    score2_tp3 = (paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[0] * paras[1]) * s.score_push
    score_tp3 = score1_tp3 + score2_tp3
    penalty_tp3 = (1 - paras[0] * paras[1] * paras[2] * paras[3] * paras[0] * paras[1] * paras[0] * paras[
        1]) * s.penalty_value
    utility_tp3 = score_tp3 - cost_tp3 - penalty_tp3

    # if task is tp4: 3, 3, 2, then its utility is:
    cost_tp4 = s.cost_place * 8. + s.cost_push * 3. + s.cost_pick * 0.
    score1_tp4 = (paras[0] + paras[1] + paras[2] + paras[0] + paras[1] + paras[2] + paras[0] + paras[1]) * s.score_place
    score2_tp4 = (paras[0] * paras[1] * paras[2] * paras[0] * paras[1] * paras[2] * paras[0] * paras[1]) * s.score_push
    score_tp4 = score1_tp4 + score2_tp4
    penalty_tp4 = (1 - paras[0] * paras[1] * paras[2] * paras[0] * paras[1] * paras[2] * paras[0] * paras[
        1]) * s.penalty_value
    utility_tp4 = score_tp4 - cost_tp4 - penalty_tp4

    # if task is tp5: 5, 2, 1, then its utility is:
    cost_tp5 = s.cost_place * 7. + s.cost_push * 2. + s.cost_pick * 1.
    score1_tp5 = (paras[0] + paras[1] + paras[2] + paras[3] + paras[4] + paras[0] + paras[1] + paras[0]) * s.score_place
    score2_tp5 = (paras[0] * paras[1] * paras[2] * paras[3] * paras[4] * paras[0] * paras[1] * paras[0]) * s.score_push
    score_tp5 = score1_tp5 + score2_tp5
    penalty_tp5 = (1 - paras[0] * paras[1] * paras[2] * paras[3] * paras[4] * paras[0] * paras[1] * paras[
        0]) * s.penalty_value
    utility_tp5 = score_tp5 - cost_tp5 - penalty_tp5

    utility_tp_s = np.array([utility_tp0, utility_tp1, utility_tp2, utility_tp3, utility_tp4, utility_tp5])
    np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
    explore_tp = np.random.choice(2, p=[s.exploration_plan, 1. - s.exploration_plan])
    max_index = np.where(utility_tp_s == np.max(utility_tp_s))
    if explore_tp == 0:
        # ------------------------------
        # print('in func_select_optimal_tp')
        # print('utility_tp_s:', utility_tp_s)
        # print('tp (optimal) is', max_index[0][0])
        np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
        tp = np.random.choice(max_index[0])
        # fidout1.write('tp (optimal) is %d\n' % tp)
        # fidout1.flush()
        # ------------------------------
        return tp
    else:
        # try to explore other space
        # np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
        # tp = np.random.choice(np.arange(0, 6, 1))
        tp = 0
        # ------------------------------
        # print('in func_select_optimal_tp')
        # print('utility_tp_s:', utility_tp_s)
        # print('tp (random) is', max_index[0][0])
        # fidout1.write('tp (random) is %d\n' % tp)
        # fidout1.flush()
        # ------------------------------
        return tp


# ----------------------------------------------------------------------------------------------------------------------
def func_perform_tp_in_sim_world(tp, world, primitive_succ_lib):
    # find the corresponding probabilities
    min_value = 999
    min_index = 999
    for experience_index in range(np.size(learnt_exp, 0)):
        a = abs(world['width'] - learnt_exp[experience_index][0])
        b = abs(world['height'] - learnt_exp[experience_index][1])
        c = abs(world['density'] - learnt_exp[experience_index][4])
        if a * a + b * b + 999 * c <= min_value:
            min_value = a * a + b * b + 999 * c
            min_index = experience_index
        else:
            continue
    prob_obj = np.array(learnt_exp[min_index][5:10])

    # ---------------------------------
    # print('world height lowest is', round(world['height'] * s.gammar1, 3))
    # print('world height lowest is', round(world['height'], 3))
    # print('primitive_y is', round(world['primitive_y'], 3))
    # ---------------------------------

    if round(world['height'] * s.gammar1, 3) <= round(world['primitive_y'], 3) <= round(world['height'],
                                                                                        3):  # primitive successfully
        # add the primitive in the library
        key = [world['width'], world['height'], world['density']]
        key = tuple(key)
        value = round(world['primitive_y'], 2)

        # ---------------------------------
        # print('key in primitive_succ_lib:', key)
        # print('value in primitive_succ_lib:', value)
        # ---------------------------------

        if key in primitive_succ_lib:
            if value not in primitive_succ_lib[key]:
                primitive_succ_lib[key].append(value)
        else:
            primitive_succ_lib[key] = [round(value, 2)]
        # perform tp and obtain results according to learnt probability
        if tp == 0:
            index_list = [0, 1, 2, 3, 4, 0, 1, 2]
        elif tp == 1:
            index_list = [0, 1, 2, 3, 0, 1, 2, 3]
        elif tp == 2:
            index_list = [0, 1, 2, 3, 0, 1, 2, 0]
        elif tp == 3:
            index_list = [0, 1, 2, 3, 0, 1, 0, 1]
        elif tp == 4:
            index_list = [0, 1, 2, 0, 1, 2, 0, 1]
        elif tp == 5:
            index_list = [0, 1, 2, 3, 4, 0, 1, 0]
        step_result_s = []
        for step in range(8):
            np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
            step_result = np.random.choice(2, p=[1. - round(prob_obj[index_list[step]], 2),
                                                 round(prob_obj[index_list[step]], 2)])
            step_result_s.append(step_result)
        world['res_step_1'] = step_result_s[0]
        world['res_step_2'] = step_result_s[1]
        world['res_step_3'] = step_result_s[2]
        world['res_step_4'] = step_result_s[3]
        world['res_step_5'] = step_result_s[4]
        world['res_step_6'] = step_result_s[5]
        world['res_step_7'] = step_result_s[6]
        world['res_step_8'] = step_result_s[7]
    else:  # grasp fails
        world['res_step_1'] = 999
        world['res_step_2'] = 999
        world['res_step_3'] = 999
        world['res_step_4'] = 999
        world['res_step_5'] = 999
        world['res_step_6'] = 999
        world['res_step_7'] = 999
        world['res_step_8'] = 999


# ----------------------------------------------------------------------------------------------------------------------
def func_perform_tp_in_real_world(tp, world):
    prob_obj = np.array([s.first_prob, s.second_prob, s.third_prob, s.fourth_prob, s.fifth_prob])

    # -------------------------------
    np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
    temp_primitive_y = round(world['primitive_y'], 3)
    # print('temp_primitive_y', temp_primitive_y)
    #  + round(np.random.normal(0, .2, 1), 2)
    # -------------------------------

    if round(world['height'] * s.gammar1, 3) <= temp_primitive_y <= round(world['height'], 3):  # primitive successfully
        if tp == 0:
            index_list = [0, 1, 2, 3, 4, 0, 1, 2]
        elif tp == 1:
            index_list = [0, 1, 2, 3, 0, 1, 2, 3]
        elif tp == 2:
            index_list = [0, 1, 2, 3, 0, 1, 2, 0]
        elif tp == 3:
            index_list = [0, 1, 2, 3, 0, 1, 0, 1]
        elif tp == 4:
            index_list = [0, 1, 2, 0, 1, 2, 0, 1]
        elif tp == 5:
            index_list = [0, 1, 2, 3, 4, 0, 1, 0]
        step_result_s = np.array([])
        # ------------------------------
        # fidout1.write('tp is %d and probability: ' % tp)
        # ------------------------------
        for step in range(8):
            # ------------------------------
            # fidout1.write('%.2f ' % prob_obj[index_list[step]])
            # fidout1.flush()
            # ------------------------------
            np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
            step_result = np.random.choice(2, p=[1. - prob_obj[index_list[step]], prob_obj[index_list[step]]])
            step_result_s = np.append(step_result_s, step_result)
        # fidout1.write('\n')
        world['res_step_1'] = step_result_s[0]
        world['res_step_2'] = step_result_s[1]
        world['res_step_3'] = step_result_s[2]
        world['res_step_4'] = step_result_s[3]
        world['res_step_5'] = step_result_s[4]
        world['res_step_6'] = step_result_s[5]
        world['res_step_7'] = step_result_s[6]
        world['res_step_8'] = step_result_s[7]
    else:  # grasp fails
        world['res_step_1'] = 999
        world['res_step_2'] = 999
        world['res_step_3'] = 999
        world['res_step_4'] = 999
        world['res_step_5'] = 999
        world['res_step_6'] = 999
        world['res_step_7'] = 999
        world['res_step_8'] = 999


# ----------------------------------------------------------------------------------------------------------------------
# def func_compute_primitive_for_real_world(width_sim_world, height_sim_world, density_sim_world, primitive_y_sim_world):
#     Dict = {}
#     for index in range(s.N):
#         temp = [width_sim_world[index], height_sim_world[index], density_sim_world[index]]
#         temp = tuple(temp)
#         if tuple(temp) in Dict:
#             Dict[tuple(temp)] = Dict[tuple(temp)] + 1
#         else:
#             Dict[tuple(temp)] = 1
#     output = sorted(Dict.items(), key=lambda e: e[1], reverse=True)
#     topk_width_height_sim_world = []
#     if len(output) <= s.k:
#         k_varied = len(output)
#     else:
#         k_varied = s.k
#     for i in range(k_varied):
#         if output[i][1] == 1:
#             np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
#             ii = np.random.randint(0, len(output))
#             topk_width_height_sim_world.append(output[ii][0])
#         else:
#             topk_width_height_sim_world.append(output[i][0])
#     primitive_y_sum = 0.
#     primitive_y_num = 0.
#     for item in topk_width_height_sim_world:
#         for index in range(s.N):
#             if item[0] == width_sim_world[index] and item[1] == height_sim_world[index]:
#                 primitive_y_sum = primitive_y_sum + primitive_y_sim_world[index]
#                 primitive_y_num = primitive_y_num + 1
#     primitive_y_real = round(primitive_y_sum / primitive_y_num, 2)
#     # print('------------------------------')
#     # print('in func_compute_primitive_for_real_world')
#     # print('sorted output:', output)
#     print('topk_width_height_sim_world:', topk_width_height_sim_world)
#     # print('------------------------------')
#     return primitive_y_real, topk_width_height_sim_world

def func_compute_primitive_for_real_world(width_sim_world, height_sim_world, density_sim_world, primitive_y_sim_world):
    Dict = {}
    for index in range(s.N):
        temp = [width_sim_world[index], height_sim_world[index], density_sim_world[index]]
        temp = tuple(temp)
        if tuple(temp) in Dict:
            Dict[tuple(temp)] = Dict[tuple(temp)] + 1
        else:
            Dict[tuple(temp)] = 1
    output = sorted(Dict.items(), key=lambda e: e[1], reverse=True)
    topk_width_height_sim_world = []
    k_temp = 0
    while k_temp <= s.k:
        for item in output:
            for num_in_item in range(item[1]):
                topk_width_height_sim_world.append(item[0])
                k_temp = k_temp + 1
                if k_temp >= s.k:
                    break
            if k_temp >= s.k:
                break
    primitive_y_sum = 0.
    primitive_y_num = 0.
    for item in topk_width_height_sim_world:
        for index in range(s.N):
            if item[0] == width_sim_world[index] and item[1] == height_sim_world[index]:
                primitive_y_sum = primitive_y_sum + primitive_y_sim_world[index]
                primitive_y_num = primitive_y_num + 1
    primitive_y_real = round(primitive_y_sum / primitive_y_num, 2)
    # print('------------------------------')
    # print('in func_compute_primitive_for_real_world')
    # print('sorted output:', output)
    # print('topk_width_height_sim_world:', topk_width_height_sim_world)
    # print('------------------------------')
    return primitive_y_real, topk_width_height_sim_world

# ----------------------------------------------------------------------------------------------------------------------
def func_compute_result_in_real_world(tp, world):
    # score for successfully placing
    if world['res_step_1'] != 999:  # grasp successfully
        score_part1 = (world['res_step_1'] + world['res_step_2'] + world['res_step_3'] + world[
            'res_step_4'] + world['res_step_5'] + world['res_step_6'] + world['res_step_7'] + world[
                           'res_step_8']) * s.score_place
    else:
        score_part1 = 0.
    # score for successful task completion
    if world['res_step_1'] == 1 and world['res_step_2'] and world['res_step_3'] == 1 and world[
        'res_step_4'] == 1 and world['res_step_5'] == 1 and world['res_step_6'] == 1 and world[
        'res_step_7'] and world['res_step_8'] == 1:  # big bonus
        score_part2 = s.score_push
        penalty = 0.
    else:
        score_part2 = 0.
        penalty = s.penalty_value
    score = score_part1 + score_part2
    # cost for different plan
    try:
        if tp == 0 or tp == 1:
            cost = s.cost_place * 8. + s.cost_push * 2. + s.cost_pick * 0.
        elif tp == 3 or tp == 4:
            cost = s.cost_place * 8. + s.cost_push * 3. + s.cost_pick * 0.
        elif tp == 5 or tp == 2:
            cost = s.cost_place * 7. + s.cost_push * 2. + s.cost_pick * 1.
    except:
        print('Not find tp!')
    result = score - cost - penalty  # the final result
    return result, score, -cost, -penalty


# ----------------------------------------------------------------------------------------------------------------------
def func_compute_succ_rate(tp, world):
    result = np.array(
        [world['res_step_1'], world['res_step_2'], world['res_step_3'], world['res_step_4'], world['res_step_5'],
         world['res_step_6'], world['res_step_7'], world['res_step_8']])
    if tp == 0:
        index_list = [0, 1, 2, 3, 4, 0, 1, 2]
    elif tp == 1:
        index_list = [0, 1, 2, 3, 0, 1, 2, 3]
    elif tp == 2:
        index_list = [0, 1, 2, 3, 0, 1, 2, 0]
    elif tp == 3:
        index_list = [0, 1, 2, 3, 0, 1, 0, 1]
    elif tp == 4:
        index_list = [0, 1, 2, 0, 1, 2, 0, 1]
    elif tp == 5:
        index_list = [0, 1, 2, 3, 4, 0, 1, 0]
    succ_diff_obj = [0] * 5
    fail_diff_obj = [0] * 5
    for i in range(8):
        if result[i] == 1:
            succ_diff_obj[index_list[i]] += 1.
        elif result[i] == 0:
            fail_diff_obj[index_list[i]] += 1.
    # ---------------------------------------------------------------
    return succ_diff_obj, fail_diff_obj


# ----------------------------------------------------------------------------------------------------------------------
def func_update_attribute(tp, width_list, height_list, friction_list, density_list, observe_in_out_Z):
    """
    step5: update configuration learner
    square_radius_list: the current square size in the simulated worlds
    square_in_out: the observation of the simulated worlds
    """
    # print('observe_in_out_Z:')
    # print(observe_in_out_Z)
    difference = np.zeros(s.N + 1, dtype=float)  # initialization
    height_list = np.array(height_list)
    if tp == 0:
        coefficient_base = [1., 2., 3., 4., 5., 1., 2., 3.]
    elif tp == 1:
        coefficient_base = [1., 2., 3., 4., 1., 2., 3., 4.]
    elif tp == 2:
        coefficient_base = [1., 2., 3., 4., 1., 2., 3., 1.]
    elif tp == 3:
        coefficient_base = [1., 2., 3., 4., 1., 2., 1., 2.]
    elif tp == 4:
        coefficient_base = [1., 2., 3., 1., 2., 3., 1., 2.]
    elif tp == 5:
        coefficient_base = [1., 2., 3., 4., 5., 1., 2., 1.]

    for x in range(s.N + 1):  # compute the observation difference between real and sim worlds
        difference_part1 = 0.
        for y in range(len(coefficient_base)):
            difference_part1 = difference_part1 + coefficient_base[y] * abs(
                observe_in_out_Z[x][y] - observe_in_out_Z[s.N][y])
        difference[x] = s.step_num * 1. - difference_part1
    sum_value = np.sum(difference)  # regularization
    difference = difference / sum_value
    # ------------------------------
    # print('difference:', difference)
    # fidout1.write('------------------------------\n')
    # fidout1.write('difference: \n')
    # fidout1.flush()
    # for item in difference:
    #     fidout1.write('%.2f  ' % item)
    #     fidout1.flush()
    # fidout1.write('\n')
    # fidout1.flush()
    # ------------------------------
    particles1 = np.array(width_list[0:-1])
    particles2 = np.array(height_list[0:-1])
    particles3 = np.array(friction_list[0:-1])
    particles4 = np.array(density_list[0:-1])
    weights = difference[0:-1]
    cumulative_sum = np.cumsum(weights)
    a = np.linspace(0., cumulative_sum[-1], len(particles1) + 2)
    indexes = np.searchsorted(cumulative_sum, a[1:-1])
    width_list[0:-1] = particles1[indexes]
    height_list[0:-1] = particles2[indexes]
    friction_list[0:-1] = particles3[indexes]
    density_list[0:-1] = particles4[indexes]
    return width_list, height_list, friction_list, density_list, difference, cumulative_sum


def penalty_size_check(new_height_list):
    result = Counter(new_height_list)
    max_count = 0
    for key in result.keys():
        if result[key] >= max_count:
            max_count = result[key]
    max_probability = round(max_count / float(len(new_height_list)), 2)
    return max_probability


# ----------------------------------------------------------------------------------------------------------------------
# record the whole running process
fidout1 = open(s.name1, 'w')
# record results acquired from real world
fidout2 = open(s.name2, 'w')
# store primitive parameters
primitive_succ_lib = {}
# record success or failure cases
succ_num_with_diff_objs = [0] * 5
fail_num_with_diff_objs = [0] * 5
succ_rate_with_diff_objs = [0] * 5
# record the optimal task plan in the whole process
tp_s = []
# record the result in the real world
result_for_plot = []
# load trained probability
learnt_exp = np.loadtxt("learntExperience_simplified_sorted.txt", dtype=float)
# name of real world
world_real = 'world_' + str(s.N)
# ----------------------------------------------------------------------------------------------------------------------
# initialize simulated world and real world
attribute1 = np.append(np.arange(1.0, 5.4, 0.4), 3.2)  # [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.2, 3.4, 3.8, 4.2, 4.6, 5.0]
attribute2 = np.append(np.arange(1.0, 5.4, 0.4), 3.2)
attribute3 = np.array([0.1, 0.4])
attribute_temp = []
for item1 in attribute1:
    for item2 in attribute2:
        for item3 in attribute3:
            if round(item1, 2) <= round(item2, 2):
                attribute_temp.append([round(item1, 2), round(item2, 2), round(item3, 2)])

try:
    np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
    selected_attributes = random.sample(attribute_temp, s.N)
except:
    print('please select a smaller N!')

for i in range(s.N + 1):
    world_name = 'world_' + str(i)
    globals()[world_name] = {}
    if i < s.N:
        globals()[world_name]['width'] = selected_attributes[i][0]
        globals()[world_name]['height'] = selected_attributes[i][1]
        globals()[world_name]['friction'] = 0.1
        globals()[world_name]['density'] = selected_attributes[i][2]
    elif i == s.N:
        globals()[world_real]['width'] = s.real_width
        globals()[world_real]['height'] = s.real_height
        globals()[world_real]['friction'] = s.real_friction
        globals()[world_real]['density'] = s.real_density
    globals()[world_name]['primitive_y'] = 999.
    globals()[world_name]['res_step_1'] = 999
    globals()[world_name]['res_step_2'] = 999
    globals()[world_name]['res_step_3'] = 999
    globals()[world_name]['res_step_4'] = 999
    globals()[world_name]['res_step_5'] = 999
    globals()[world_name]['res_step_6'] = 999
    globals()[world_name]['res_step_7'] = 999
    globals()[world_name]['res_step_8'] = 999
    # ------------------------------
    # fidout1.write('world: %d, width: %.2f, height: %.2f, density: %.2f\n' % (
    #     i, globals()[world_name]['width'], globals()[world_name]['height'],
    #     globals()[world_name]['density']))
    # fidout1.flush()
# ----------------------------------------------------------------------------------------------------------------------
iter = 0
Z_num = 0
while iter <= s.M:
    np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
    # compute the optimal tp; store tp; record tp
    tp = func_select_optimal_tp(succ_rate_with_diff_objs)
    tp_s.append(tp)
    # record the observation Z iters
    observe_in_out_Z = np.zeros((s.N + 1, 8))
    while Z_num <= s.Z:  # perform tp for repeat times
        iter = iter + 1
        # ------------------------------
        print('iter: ', iter)
        fidout1.write('iter: %d\n' % iter)
        fidout1.flush()
        # ------------------------------
        # perform tp in simulated worlds
        width_sim_world = []
        height_sim_world = []
        density_sim_world = []
        primitive_y_sim_world = []
        for index in range(s.N):
            world_sim = 'world_' + str(index)
            func_perform_tp_in_sim_world(tp, globals()[world_sim], primitive_succ_lib)

            width_sim_world.append(round(globals()[world_sim]['width'], 2))
            height_sim_world.append(round(globals()[world_sim]['height'], 2))
            density_sim_world.append(round(globals()[world_sim]['density'], 2))
            primitive_y_sim_world.append(round(globals()[world_sim]['primitive_y'], 2))

        # print('primitive_succ_lib:', primitive_succ_lib)
        # compute primitive for the real world
        primitive_y_real, topk_width_height_sim_world = func_compute_primitive_for_real_world(width_sim_world,
                                                                                              height_sim_world,
                                                                                              density_sim_world,
                                                                                              primitive_y_sim_world)
        # check which case1, case2, case3?
        temp_data = []
        for index in range(s.N + 1):
            world_name = 'world_' + str(index)
            if globals()[world_name]['res_step_1'] == 999:
                temp_data.append(999)
            else:
                temp_data.append(1)
        if any(ii == 999 for ii in temp_data[0:-1]):
            is_case1 = 1
            is_case2 = 999
            is_case3 = 999
            # ------------------------------
            # fidout1.write('is_case1, particles NO and real world ?\n')
            fidout1.write('case1\n')
            fidout1.flush()
            # ------------------------------
        else:
            is_case1 = 999
            if temp_data[-1] == 999:
                is_case2 = 1
                is_case3 = 999
                # ------------------------------
                # fidout1.write('is_case2, particles YES and real world NO\n')
                fidout1.write('case2\n')
                fidout1.flush()
                # ------------------------------
            else:
                is_case3 = 1
                is_case2 = 999
                # ------------------------------
                # fidout1.write('is_case3, particles YES and real world YES\n')
                fidout1.write('case3\n')
                fidout1.flush()
                # ------------------------------

        if is_case1 == 1:
            #globals()[world_real]['primitive_y'] = primitive_y_real
            globals()[world_real]['primitive_y'] = 999.

        if is_case2 == 1:
            np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
            primitive_y_real = primitive_y_real + round(np.random.normal(0, s.max_square, 1), 2)
            globals()[world_real]['primitive_y'] = primitive_y_real

        if is_case3 == 1:
            np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
            exploration_signal1 = np.random.choice(2, p=[s.exploration_primitive, 1. - s.exploration_primitive])
            if exploration_signal1 == 0:
                globals()[world_real]['primitive_y'] = primitive_y_real
            # else:
            #     # ------------------------------
                # print('Primitive Not Change in real world!')
                # fidout1.write('Primitive Not Change in real world!\n')
                # fidout1.flush()
                # ------------------------------
        # ------------------------------
        fidout1.write('tp: %d\n' % tp)
        # ------------------------------
        reward_T = []
        score_T = []
        cost_T = []
        penalty_T = []

        # ------------------------------
        # print('real world primitive:', round(globals()[world_real]['primitive_y'], 3))
        # fidout2.write('%.2f\n' % round(globals()[world_real]['primitive_y'], 3))
        # ------------------------------

        for test_number in range(5):
            # perform tp in the real world
            func_perform_tp_in_real_world(tp, globals()[world_real])
            # compute average score of tp
            reward_temp, score_temp, cost_temp, penalty_temp = func_compute_result_in_real_world(tp,
                                                                                                 globals()[world_real])
            # ------------------------------
            # fidout1.write('reward_temp, score_temp, cost_temp, penalty_temp: %.2f, %.2f, %.2f, %.2f\n' % (
            #     reward_temp, score_temp, cost_temp, penalty_temp))
            # ------------------------------
            reward_T.append(reward_temp)
            score_T.append(score_temp)
            cost_T.append(cost_temp)
            penalty_T.append(penalty_temp)

        reward = np.mean(reward_T)
        score = np.mean(score_T)
        cost = np.mean(cost_T)
        penalty = np.mean(penalty_T)

        result_for_plot.append(reward)
        # ------------------------------
        # fidout1.write('reward, score, cost, penalty: %.2f, %.2f, %.2f, %.2f\n' % (reward, score, cost, penalty))
        fidout1.write('reward: %.2f\n' % reward)
        fidout1.flush()
        print('reward:', reward)
        # ------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        succ_diff_obj, fail_diff_obj = func_compute_succ_rate(tp, globals()[world_real])
        for index in range(5):
            succ_num_with_diff_objs[index] = succ_num_with_diff_objs[index] + succ_diff_obj[index]
            fail_num_with_diff_objs[index] = fail_num_with_diff_objs[index] + fail_diff_obj[index]
        succ_rate_with_diff_objs = np.array(succ_num_with_diff_objs) / (
                np.array(succ_num_with_diff_objs) + np.array(fail_num_with_diff_objs) + np.array(
            [0.001, 0.001, 0.001, 0.001, 0.001]))
        succ_rate_with_diff_objs = [round(item, 2) for item in succ_rate_with_diff_objs]
        # ------------------------------
        # fidout1.write('succ_num_with_diff_objs:\n')
        # for item in succ_num_with_diff_objs:
        #     fidout1.write('item %.1f ' % item)
        # fidout1.write('\n')
        # fidout1.flush()
        # fidout1.write('fail_num_with_diff_objs:\n')
        # for item in fail_num_with_diff_objs:
        #     fidout1.write('item %.1f ' % item)
        # fidout1.write('\n')
        # fidout1.flush()
        # fidout1.write('succ_rate_with_diff_objs:\n')
        # for item in succ_rate_with_diff_objs:
        #     fidout1.write('item %.1f ' % item)
        # fidout1.write('\n')
        # fidout1.flush()
        # ------------------------------
        for index in range(s.N):  # update primitive function
            world_sim = 'world_' + str(index)
            func_update_primitive(globals()[world_sim], primitive_succ_lib)
        # ------------------------------
        # store data for updating configuration
        if Z_num == 0:  # initialize
            width_list = [0] * (s.N + 1)
            height_list = [0] * (s.N + 1)
            friction_list = [0] * (s.N + 1)
            density_list = [0] * (s.N + 1)
            for index in range(s.N + 1):
                world_name = 'world_' + str(index)
                width_list[index] = globals()[world_name]['width']
                height_list[index] = globals()[world_name]['height']
                friction_list[index] = globals()[world_name]['friction']
                density_list[index] = globals()[world_name]['density']
        # store observation each iter
        observe_in_out = []
        for index in range(s.N + 1):
            temp = []
            world_name = 'world_' + str(index)
            temp.append(globals()[world_name]['res_step_1'])
            temp.append(globals()[world_name]['res_step_2'])
            temp.append(globals()[world_name]['res_step_3'])
            temp.append(globals()[world_name]['res_step_4'])
            temp.append(globals()[world_name]['res_step_5'])
            temp.append(globals()[world_name]['res_step_6'])
            temp.append(globals()[world_name]['res_step_7'])
            temp.append(globals()[world_name]['res_step_8'])
            observe_in_out.append(temp)
        # only all cases get results --> add
        temp = []
        for index in range(s.N + 1):
            temp.append(observe_in_out[index][0])  # add the first result to temp
        if all(item <= 1 for item in temp):
            observe_in_out_Z = np.array(observe_in_out_Z) + np.array(observe_in_out)
            Z_num = Z_num + 1  # count the satisfied case in observe_in_out_Z
            # fidout1.write('add 1 in observe_in_out_Z and its num is %d\n' % Z_num)
    # update configuration
    observe_in_out_Z = observe_in_out_Z / (s.Z + 1.)
    observe_in_out_Z = np.round(observe_in_out_Z, 3)
    new_width_list, new_height_list, new_friction_list, new_density_list, diff, sum = func_update_attribute(tp,
                                                                                                            width_list,
                                                                                                            height_list,
                                                                                                            friction_list,
                                                                                                            density_list,
                                                                                                            observe_in_out_Z)
    # re-initialize
    observe_in_out_Z = []
    Z_num = 0
    # ------------------------------
    # fidout1.write('Attributes are updated!\n')
    # fidout1.write('new_width_list, new_height_list, new_density_list are: \n')
    # for index in range(s.N + 1):
    #     fidout1.write('(index: %d, width:%.2f, height:%.2f, density:%.2f)\n' % (
    #     index, new_width_list[index], new_height_list[index], new_density_list[index]))
    # fidout1.write('\n')
    # fidout1.flush()
    # ------------------------------
    # replace old attributes with new ones
    for i in range(s.N):
        world_sim = 'world_' + str(i)
        globals()[world_sim]['width'] = new_width_list[i]
        globals()[world_sim]['height'] = new_height_list[i]
        globals()[world_sim]['friction'] = new_friction_list[i]
        globals()[world_sim]['density'] = new_density_list[i]
        # because the attributes have been changed, thus update its primitive parameters
        key = [globals()[world_sim]['width'], globals()[world_sim]['height'], globals()[world_sim]['density']]
        try:
            values = primitive_succ_lib[tuple(key)]
            np.random.seed(int(time.time() * 1e4 - 1.6251 * 1e13))
            value = random.choice(values)
            globals()[world_sim]['primitive_y'] = round(value, 2)
        except:
            print('not find key in primitive_succ_lib')
    # ------------------------------
    # for index in range(s.N):  # update primitive function
    #     world_sim = 'world_' + str(index)
    #     func_update_primitive(globals()[world_sim], primitive_succ_lib)
    # ------------------------------
# write results
for iter in range(s.M):
    fidout2.write(str(result_for_plot[iter]))
    fidout2.write(' ')
fidout2.write('\n')
fidout1.close()
fidout2.close()

end = time.time()
print('running time:', end - start)
print('current folder:', os.getcwd())
