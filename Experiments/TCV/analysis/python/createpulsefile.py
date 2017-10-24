import topic21Mds

shotL = (51080, 51084, 51124, 51126, 51130, 51132, 51133, 51134, 51138, 51139,
         51141, 51176, 51358, 52062, 52065, 52212, 52629, 52633, 52644, 52650,
         52766, 52767, 53514, 53516, 53518, 53520, 53562, 53564, 53565, 53569,
         53571, 53573, 53575, 53577, 53579, 53582, 54867, 54868,
         54869, 54870, 54873, 54874, 54876)
for shot in shotL:
    try:
        Out = topic21Mds.Tree(shot)
        Out.toMds()
        print('Written pulse file for shot %5i' % shot)
    except:
        print('Pulse file not written for shot %5i' % shot)
