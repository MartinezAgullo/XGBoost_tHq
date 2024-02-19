##############
#
#
#############

import os
import ROOT

########## 
# main
########## 
def main():
    # Specify the root folder to start the search
    output_folder = "/lustre/ific.uv.es/grid/atlas/t3/pamarag/tHq_analysis/13TeV/NewSamples_FromBonnOS_Valencia_Structure_BDT_tHq_ttbar/"

    print("Listing problematic samples")
    for systematic in os.listdir(output_folder):
        print(" Reading :: "+str(systematic))
        systematic_path = os.path.join(output_folder, systematic)
        included_already = False #Controls if the syst has been included in the list_of_bad_systematics
        for file_name in os.listdir(systematic_path):
            if file_name.endswith(".root"):
                sample_path = os.path.join(systematic_path, file_name)
                tree_name = treeNamer(systematic, file_name)
                GoodInjection_bdt_tHq, GoodInjection_bdt_ttbar = calculate_mean(sample_path, tree_name)
                if GoodInjection_bdt_tHq == False:
                    print("problem with samples " +str(systematic) + "/"+str(file_name))
                    #print("Adding BDT(tHq) score for "+str(systematic))
                    
                    






##########
# calculate_mean(): Calculates the mean and std of the bdt_tHq and bdt_ttbar for each process and systematic.
#                   The bdts to test have to be hardcoded. 
##########
def calculate_mean(file_path, tree_name):
    #print("Reading :: " + str(file_path))
    #print("Tree :: "+str(tree_name))

    GoodInjection_bdt_tHq = True
    GoodInjection_bdt_ttbar = True

    try:
        # Open the ROOT file and the TTree
        root_file = ROOT.TFile.Open(file_path)
        tree = root_file.Get(tree_name)
    
        ttbar = True

        temp_h1_tHq = ROOT.TH1F("BDT(tHq) Score", "BDT(tHq) Score", 100, 0, 1)
        if ttbar: temp_h1_ttbar = ROOT.TH1F("BDT(ttbar) Score", "BDT(ttbar) Score", 100, 0, 1)

        # Loop over the entries in the TTree
        evts = 0
        for event in tree:
            temp_h1_tHq.Fill(event.bdt_tHq)
            if ttbar: temp_h1_ttbar.Fill(event.bdt_ttbar)
            evts = evts+1
    except Exception as e:
        print("ERROR with: " + file_path)
        print(e)
        return False, False

        
    bdt_tHq_RMS = temp_h1_tHq.GetRMS() # Get Standard deviation
    bdt_tHq_mean = temp_h1_tHq.GetMean() # Get mean
    if ttbar:
        bdt_ttbar_RMS = temp_h1_ttbar.GetRMS() 


    if bdt_tHq_RMS == 0.0 and evts > 1: # The events with a single entry will have 0 std
        print("Warning for "+file_path+": ")
        print("\t The bdt_tHq was not stored properly. BDT(tHq) = "+str(bdt_tHq_mean) +" \pm "+str(bdt_tHq_RMS))
        print("\t Events in sample = "+str(evts))
        GoodInjection_bdt_tHq = False

    if ttbar:
        if bdt_ttbar_RMS== 0.0 and evts > 1:
            print("\t The bdt_ttbar was not stored properly in sample = "+str(evts))
            GoodInjection_bdt_ttbar = False

    root_file.Close()
    
    return GoodInjection_bdt_tHq, GoodInjection_bdt_ttbar

########## 
# treeNamer(): The AFII samples have different tree names
########## 
def treeNamer(systName, processName):

    tree_name = "tHqLoop_" + systName
    if "AFII" in processName:
        if systName == "JET_PunchThrough_MC16__1up_Loose": tree_name = "tHqLoop_JET_PunchThrough_AFII__1up_Loose"
        if systName == "JET_PunchThrough_MC16__1down_Loose": tree_name = "tHqLoop_JET_PunchThrough_AFII__1down_Loose"
        if systName == "JET_JER_DataVsMC_MC16__1up_PseudoData_Loose": tree_name = "tHqLoop_JET_JER_DataVsMC_AFII__1up_PseudoData_Loose"
        if systName == "JET_JER_DataVsMC_MC16__1down_PseudoData_Loose": tree_name = "tHqLoop_JET_JER_DataVsMC_AFII__1down_PseudoData_Loose"
        if systName == "JET_JER_DataVsMC_MC16__1up_Loose": tree_name = "tHqLoop_JET_JER_DataVsMC_AFII__1up_Loose"
        if systName == "JET_JER_DataVsMC_MC16__1down_Loose": tree_name = "tHqLoop_JET_JER_DataVsMC_AFII__1down_Loose"

    if systName == "alternative_sample": tree_name = "tHqLoop_nominal_Loose"
   # if systName == "alternative_sample" and "AFII" in processName: tree_name = "tHqLoop_nominal_Loose"
    return tree_name

if __name__ == '__main__':
  main()
