import topic21Mds

shotL = (61446, 61447, 61449, 61450, 61452, 61477,
         61478, 61479, 61480, 61481, 61483, 61484)
for shot in shotL:
#    try:
    Out = topic21Mds.Tree(shot, interelm=True)
    Out.toMds()
    print('Written pulse file for shot %5i' % shot)
    # except:
    #     print('Pulse file not written for shot %5i' % shot)
