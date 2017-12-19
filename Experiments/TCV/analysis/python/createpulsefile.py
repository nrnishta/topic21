import topic21Mds

shotL = (51180)
for shot in shotL:
    try:
        Out = topic21Mds.Tree(shot)
        Out.toMds()
        print('Written pulse file for shot %5i' % shot)
    except:
        print('Pulse file not written for shot %5i' % shot)
