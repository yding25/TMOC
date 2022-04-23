# ----------------------
name1 = 'log.txt'  # record the whole running process, especially some important paramters
name2 = 'result.txt'  # record results acquired from the real world
# ----------------------
N = 156  # the number of simulated worlds
M = 5000  # the number of iters
Z = 10  # Z is the repeated number; control the converge speed
# ----------------------
step_num = 8  # step number
# ----------------------
min_square = 1.
max_square = 5.
real_width = 3.2
real_height = 3.2
real_friction = 0.1
real_density = 0.4
# ----------------------
cost_place = 10.
cost_push = 30.
cost_pick = 15.
score_place = 40.
score_push = 80.
penalty_value = 50.
# ----------------------
first_prob = 1.0
second_prob = .9
third_prob = .7
fourth_prob = .4
fifth_prob = .2
# ----------------------
# exploration_plan = 0.8
# exploration_primitive = 0.8
exploration_plan = 0.8
exploration_primitive = 0.5
# ----------------------
gammar1 = 0.85
# ----------------------
k = 10
# ----------------------
max_density = 0.5
min_density = 0.1
