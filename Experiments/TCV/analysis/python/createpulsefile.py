import topic21Mds

shotL = (58608, 58610, 58611, 58614, 58623, 58624,
         58629, 58635, 58637, 58639)
for shot in shotL:
    try:
        Out = topic21Mds.Tree(shot)
        Out.toMds()
        print('Written pulse file for shot %5i' % shot)
    except:
        print('Pulse file not written for shot %5i' % shot)
