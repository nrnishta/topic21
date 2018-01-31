import topic21Mds

shotL = (52022, 52126, 52128, 52129,
         52131, 52198, 52121, 52205)
for shot in shotL:
    try:
        Out = topic21Mds.Tree(shot)
        Out.toMds()
        print('Written pulse file for shot %5i' % shot)
    except:
        print('Pulse file not written for shot %5i' % shot)
