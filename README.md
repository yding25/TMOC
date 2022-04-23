# TMOC

If you use this code, please cite our following paper

@article{ding2022learning,
  title={Learning to ground objects for robot task and motion planning},
  author={Ding, Yan and Zhang, Xiaohan and Zhan, Xingyue and Zhang, Shiqi},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={5536--5543},
  year={2022},
  publisher={IEEE}
}

## Environment
* conda create -n pybox2d python=2.7 && conda activate pybox2d
* conda install -c https://conda.anaconda.org/kne pybox2d
* pip install numpy==1.16.6
* pip install scipy==1.2.3
* pip install sklearn==0.20.4

## Task Planner
### cd task_planner
#### function
* describe the question
### blocks.lp world0.lp
#### function
* descripe goals 
#### example: run 'clingo *.lp -c n=10'

## Learning Experience
### autorun.py
#### function
* automoatically run TMPUD project

### main.py
#### function
* run our ego car following task and motion planning

### replay.py
#### function
* play recording video stored in "/saved"
