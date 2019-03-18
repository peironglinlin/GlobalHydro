#Peirong Lin (upload 2019-03-17)
#Split ~0.24 million cells into 40 jobs for sbatch

import numpy as np
import os
import fileinput
import os.path

njob = 40

for i in range(njob):
    ijob = i
    #copy new parallel submit cmd file
    oldfile = 'parallel_control.cmd'
    newfile = 'parallel_%03d'%ijob+'.cmd'
    os.system('cp '+oldfile+' '+newfile)
    
    #modify new submit script
    with fileinput.FileInput(newfile,inplace=True) as file:
        for line in file:
            if 'srun python' in line :
                print('srun python 1_main_matching.py -n '+str(njob)+' -j '+str(ijob)+' >& log.%03d'%ijob+''.strip())
            else:
                print(line.strip())

    #submit job
    print('... submitting '+newfile+' ...')
    os.system('sbatch '+newfile)
