import os, sys, time, subprocess
from subprocess import *
import numpy as np

def available_gpu_number(nsml_path):
    command_stdout = subprocess.Popen([nsml_path, 'gpubusy'], stdout=PIPE).communicate()[0]
    command_stdout = command_stdout.decode("utf-8")
    loc_is    = command_stdout.find('is')
    loc_slash = command_stdout.find('/')
    avail_gpu_num = int(command_stdout[loc_is+3:loc_slash])
    return avail_gpu_num


# check nsml_path if something's wrong...
nsml_path = '/Users/dae/nsml_client.darwin.amd64.dev/nsml'


# ############### Experiments settings
# nf = [16, 32]
# nl = [40, 120]
# md = [
#       # 'dense1',
#       # 'dense2',
#       # 'dense3',
#       # 'dense4',
#       'dense5'
#      ]
# reduc = [1/2, 1.0]
#
# for i in range(len(nf)):
#     for j in range(len(nl)):
#         for k in range(len(md)):
#             for l in range(len(reduc)):
#                 run_command = nsml_path+' run -a "--num_features {} --num_layers {} --model_name {} --reduction {}"'.format(nf[i], nl[j], md[k], reduc[l])
#                 subprocess.call (run_command, shell=True)



# Download Settings
# exps = [700,699,698,697,696,695,694,693,692,691,
#         690,689,688,687,686,685,681,676,675,674,
#         673,672,671,670,669,668,667,666,665,664,
#         663,662,661,659,658,657,652,651,650,649,
#         648,647,646,645,644,643,642,641,640,639,
#         638,637
#         ]

for i in range(821,833):
    run_command = nsml_path + ' download KR61811/None/{} ~/Desktop/exp -f /app/test_plot'.format(i)
    subprocess.call(run_command, shell=True)
    run_command = nsml_path + ' download KR61811/None/{} ~/Desktop/exp -f /app/logs'.format(i)
    subprocess.call(run_command, shell=True)

    
# # Delete Settings
# exps = [700,699,698,697,696,695,694,693,692,691,
#         690,689,688,687,686,685,681,676,675,674,
#         673,672,671,670,669,668,667,666,665,664,
#         663,662,661,659,658,657,652,651,650,649,
#         648,647,646,645,644,643,642,641,640,639,
#         638,637
#         ]
# run_command = nsml_path + ' rm '
# for i in exps:
#     run_command += 'KR61811/None/{} '.format(i)
#
# subprocess.call(run_command, shell=True)
