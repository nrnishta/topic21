# For the creation of pulse file for the TCV Topic 21 Experiment
import MDSplus as mds
shot = -1
# boolean for the creation of the pulse file
# we need to create the tags list for the
create_nodes = True
tree = mds.Tree('tcv_topic21', shot, 'NEW')
if create_nodes:
    # -------------------
    # this is for Fast Probe
    tree.setDefault(tree.addNode(".FP", "STRUCTURE"))  # U-Probe
    # --------------------
    # 1st Plunge structure
    tree.setDefault(tree.addNode(".FIRSTPLUNGE"))
    tree.addNode('.PROFILE', 'STRUCTURE')
    tree.setDefault(tree.getNode('.PROFILE'))
    tree.addNode('EN', "SIGNAL").addTag('FP_1PL_EN')
    tree.addNode('TE', "SIGNAL").addTag('FP_1PL_TE')
    tree.addNode('EN_ERR', "SIGNAL").addTag('FP_1PL_ENERR')
    tree.addNode('TE_ERR', "SIGNAL").addTag('FP_1PL_TEERR')
    tree.addNode('VFT', "SIGNAL").addTag('FP_1PL_VFT')
    tree.addNode('VFM', "SIGNAL").addTag('FP_1PL_VFM')
    tree.addNode('VFB', "SIGNAL").addTag('FP_1PL_VFB')
    tree.addNode('JS', "SIGNAL").addTag('FP_1PL_JS')
    tree.addNode('RHO', "SIGNAL").addTag('FP_1PL_RHO')
    tree.addNode('RRSEP', "SIGNAL").addTag('FP_1PL_RRSEP')

    # add now the nodes for the position as a function of time
    tree.setDefault(tree.getNode('-'))
    tree.addNode('.POSITION', 'STRUCTURE')
    tree.setDefault(tree.getNode('.POSITION'))
    tree.addNode('RHOT', "SIGNAL").addTag('FP_1PL_RHOT')
    tree.addNode('RRSEPT', "SIGNAL").addTag('FP_1PL_RRSEPT')
    # --------------------
    # 2nd Plunge structure
    tree.setDefault(tree.getNode('--'))
    tree.setDefault(tree.addNode(".SECONDPLUNGE"))
    tree.addNode('.PROFILE', 'STRUCTURE')
    tree.setDefault(tree.getNode('.PROFILE'))
    tree.addNode('EN', "SIGNAL").addTag('FP_2PL_EN')
    tree.addNode('TE', "SIGNAL").addTag('FP_2PL_TE')
    tree.addNode('EN_ERR', "SIGNAL").addTag('FP_2PL_ENERR')
    tree.addNode('TE_ERR', "SIGNAL").addTag('FP_2PL_TEERR')
    tree.addNode('VFT', "SIGNAL").addTag('FP_2PL_VFT')
    tree.addNode('VFM', "SIGNAL").addTag('FP_2PL_VFM')
    tree.addNode('VFB', "SIGNAL").addTag('FP_2PL_VFB')
    tree.addNode('JS', "SIGNAL").addTag('FP_2PL_JS')
    tree.addNode('RHO', "SIGNAL").addTag('FP_2PL_RHO')
    tree.addNode('RRSEP', "SIGNAL").addTag('FP_2PL_RRSEP')
    # add now the nodes for the position as a function of time
    tree.setDefault(tree.getNode('-'))
    tree.addNode('.POSITION', 'STRUCTURE')
    tree.setDefault(tree.getNode('.POSITION'))
    tree.addNode('RHOT', "SIGNAL").addTag('FP_2PL_RHOT')
    tree.addNode('RRSEPT', "SIGNAL").addTag('FP_2PL_RRSEPT')

    # ----------------------
    # Now the Lambda
    tree.setDefault(tree.getNode('---'))
    tree.setDefault(tree.addNode(".LAMBDA"))
    tree.addNode('DIVU', "SIGNAL").addTag('LDIVU')
    tree.addNode('DIVX', "SIGNAL").addTag('LDIVX')
    tree.addNode('RHO', "SIGNAL").addTag('LRHO')
    tree.setDefault(tree.getNode('-'))
    # now save the parallel connection length
    tree.setDefault(tree.addNode('.LPARALLEL'))
    tree.addNode('DIVU', "SIGNAL").addTag('LPDIVU')
    tree.addNode('DIVX', "SIGNAL").addTag('LPDIVX')
    tree.addNode('RHO', "SIGNAL").addTag('LPRHO')
    # -----------------------
    # Now the interELM profiles for H-Mode
    tree.setDefault(tree.getNode('-'))
    tree.setDefault(tree.addNode(".LPNOELM"))
    tree.addNode('DENS',"SIGNAL").addTag('LANGNE')
    tree.addNode('TE',"SIGNAL").addTag('LANGTE')
    tree.addNode('JSAT2',"SIGNAL").addTag('LANGJSAT2')
    tree.addNode('P_PERP',"SIGNAL").addTag('LANGPPERP')
    tree.addNode('POS',"NUMERIC").addTag('LANGPOS')
    tree.addNode('AREA',"SIGNAL").addTag('LANGAREA')
    tree.addNode('PROBES',"NUMERIC").addTag('LPROBES')
    tree.addNode('RHO_PSI',"SIGNAL").addTag('LANGRHO')
    tree.addNode('RHO_PSI2',"SIGNAL").addTag('LANGRHO2')
    tree.addNode('DSEP_MID',"SIGNAL").addTag('LANGDSEP')
    tree.addNode('DSEP_MID2',"SIGNAL").addTag('LANGDSEP2')
    tree.addNode('TIME',"NUMERIC").addTag('LANGTIME')
    tree.addNode('TIME2',"NUMERIC").addTag('LANGTIME2')

    tree.write()

