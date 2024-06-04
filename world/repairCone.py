#!/usr/bin/python3
import re
import numpy as np

myregex=r"<link name='cone_([0-9]+)_link'>"
myregPose = r"<pose>(.*)</pose>"
validPose=''

poseOnly = True

with open('track_cones.sdf') as f:
    while True:
        line = f.readline()
        if not line:
            break
        lastPose = re.search(myregPose,line)
        if lastPose is not None:
            validPose = lastPose.group(0)
            validPoseCoord = lastPose.group(1)
        m = re.search(myregex, line)
        if m is not None:
            mygroup = m.group(1)
            if (int(mygroup))>=1:
                if poseOnly:
                    tp = validPoseCoord.split(' ')
                    print(', '.join(tp[:3]))
                    continue
                else:
                    print(f"""
        <static>true</static>
        {validPose}
        <enable_wind>false</enable_wind>
    </model>
            
    <model name='cones_{mygroup}'>
    """)
        if not poseOnly:
            print(line, end='')
