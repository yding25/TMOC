import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
import numpy as np
import time
import active_learners.helper as helper
from kitchen2d.kitchen_constants import *

# key size
container_width = 8.
container_height = 6.
container_thick = 0.5

# key positions
container_pos = (20., 20.)
dest = (40., 5.)
init_gripper_pos = (-30., 15.)

pos1 = (-50., 0.)
pos2 = (-40., 0.)
pos3 = (-30., 0.)
pos4 = (-20., 0.)
pos5 = (-10., 0.)

horizon_x = 2.
container_pos_left = container_pos[0] - 5
container_pos_right = container_pos[0] + 5
ungrasp_height = container_height + 0.6 * 20

def create_world():
    # fixed kitchen world
    kitchen = Kitchen2D(**SETTING)
    kitchen.enable_gui()
    #kitchen.disable_gui()
    container = ks.make_cup(kitchen, container_pos, 0, container_width, container_height, container_thick)
    gripper = Gripper(kitchen, init_gripper_pos, 0)
    return kitchen, container, gripper

def create_container():
    container = ks.make_cup(kitchen, container_pos, 0, container_width, container_height, container_thick)
    return container

def create_square(kitchen, square_pos, width, height):
    square = ks.make_block(kitchen, square_pos, 0, width, height)
    return square

def move_square(container, gripper, square):
    # grasp square
    gripper.find_path((square.position[0], ungrasp_height), 0, maxspeed=MAX_SPEED)
    gripper.grasp(square, pos_ratio_square)

    gripper.find_path((square.position[0], ungrasp_height), 0, maxspeed=MAX_SPEED)
    gripper.find_path((container.position[0], ungrasp_height), 0, maxspeed=MAX_SPEED)

    # move above the container
    result = gripper.place((container.position[0] + np.random.uniform(-2.5, 2.5), ungrasp_height), 0)

    # push container
    rel_pos = (horizon_x, pos_ratio_push_container)
    gripper.push(container, rel_pos, container_pos_left, MAX_SPEED)

    gripper.find_path((container.position[0], ungrasp_height), 0, maxspeed=MAX_SPEED)
    gripper.push(container, rel_pos, container_pos_right, MAX_SPEED)
    gripper.find_path((init_gripper_pos[1], ungrasp_height), 0, maxspeed=MAX_SPEED)


def push_container(container, gripper):
    # push container
    rel_pos = (horizon_x, pos_ratio_push_container)
    gripper.push(container, rel_pos, dest[0], MAX_SPEED * 2)


def in_container(container, item):
    ppos = item.position - container.position
    if (container.usr_w - container_thick) / 2. >= abs(ppos[0]):
        inContainer = 1
    else:
        inContainer = 0
    return inContainer


def destroy(kitchen, item):
    kitchen.world.DestroyBody(item)


def main(width, height, primitiveParameter):
    ''' adjustable parameters '''
    global gripper, container, kitchen
    global pos_ratio_square, pos_ratio_push_container

    # when 0.1 means the primitive will certainly succeed
    # ratio for grasp square
    pos_square = primitiveParameter * height
    pos_ratio_square = pos_square / height

    # ratio for push contianer
    pos_push_container = primitiveParameter * container_height
    pos_ratio_push_container = pos_push_container / container_height

    kitchen, container, gripper = create_world()

    square1 = create_square(kitchen, pos1, width, height)
    square2 = create_square(kitchen, pos2, width, height)
    square3 = create_square(kitchen, pos3, width, height)
    square4 = create_square(kitchen, pos4, width, height)
    square5 = create_square(kitchen, pos5, width, height)

    print('---- before push ----')

    move_square(container, gripper, square1)
    move_square(container, gripper, square2)
    move_square(container, gripper, square3)
    move_square(container, gripper, square4)
    move_square(container, gripper, square5)

    push_container(container, gripper)

    print('---- after push ----')

    fileout.write('%d ' % in_container(container, square1))
    fileout.flush()
    fileout.write('%d ' % in_container(container, square2))
    fileout.flush()
    fileout.write('%d ' % in_container(container, square3))
    fileout.flush()
    fileout.write('%d ' % in_container(container, square4))
    fileout.flush()
    fileout.write('%d ' % in_container(container, square5))
    fileout.flush()

    fileout.write('\n')
    time.sleep(0.1)


if __name__ == '__main__':
    fileout1 = open('learntExperience0.txt', 'w')

    for width in [2.8]:
        for heigth in [2.8]:
            for primitiveParameter in [0.1]:
                for friction in [0.1]:
                    for density in [0.4]:
                        
                        fidin = open('./kitchen2d/setting.py', 'w')
                        fidin.write('friction = %.1f\n' % friction)
                        fidin.flush()
                        fidin.write('density = %.1f\n' % density)
                        fidin.flush()
                        fidin.write('OPEN_WIDTH = %.1f\n' % (width + 3.0))
                        fidin.flush()
                        fidin.close()

                        fileout = open('temp_experience0.txt', 'w')

                        for i in range(30):
                            try:
                                main(width, heigth, primitiveParameter)
                            except AssertionError:
                                print('exception')
                                fileout.write('%d ' % 0)
                                fileout.flush()
                                fileout.write('%d ' % 0)
                                fileout.flush()
                                fileout.write('%d ' % 0)
                                fileout.flush()
                                fileout.write('%d ' % 0)
                                fileout.flush()
                                fileout.write('%d ' % 0)
                                fileout.flush()
                                fileout.write('\n')

                        X = np.loadtxt("temp_experience0.txt", dtype=float)  # read txt data

                        fileout.close()

                        [first_true_probability, second_true_probability, third_true_probability, fourth_true_probability, fifth_true_probability] = np.mean(X, axis=0)

                        fileout1.write('%0.1f %0.1f %0.1f %0.1f %0.1f %0.2f %0.2f %0.2f %0.2f %0.2f' % (
                        width, heigth, primitiveParameter, friction, density, first_true_probability, second_true_probability, third_true_probability, fourth_true_probability,
                        fifth_true_probability))
                        fileout1.write('\n')
                        fileout1.flush()

    fileout1.close()
