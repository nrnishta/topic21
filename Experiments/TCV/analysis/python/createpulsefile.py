import topic21Mds

shotL = (57418, 57425, 57437, 57450, 57454, 57459, 57461, 57497)
for shot in shotL:
    Out = topic21Mds.Tree(shot)
    Out.toMds()
    print('Written pulse file for shot %5i' % shot)
