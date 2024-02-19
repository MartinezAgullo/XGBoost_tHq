import yaml
###########################################################
#   Create a table with the variables used in the BDTs    #
###########################################################


def main():
    bdt_tHq_SS = read_yaml('/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_SS.yaml', 'ShortMVAtHq')
    bdt_tHq_OS = read_yaml('/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml', 'ShortMVAtHq')
    bdt_ttbar_OS = read_yaml('/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_OS.yaml', 'ShortMVAttbar')

    label_names_ss = extract_label_names(bdt_tHq_SS)
    label_names_os_hq = extract_label_names(bdt_tHq_OS)
    label_names_os_tbar = extract_label_names(bdt_ttbar_OS)

    combined_variables = set(label_names_ss) | set(label_names_os_hq) | set(label_names_os_tbar)

    bdt_presence = {var: {'isInbdt_tHq_SS': var in label_names_ss,
                          'isInbdt_tHq_OS': var in label_names_os_hq,
                          'isInbdt_ttbar_OS': var in label_names_os_tbar}
                    for var in combined_variables}

    print_latex_format(bdt_presence)

    #combined_label_names = list(set(label_names_ss + label_names_os_hq + label_names_os_tbar))    
    #print_latex_format(combined_label_names)


def read_yaml(file_name, key):
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
        return data.get(key, {})

def extract_label_names(data):
    label_names = []
    for variable in data.values():
        label_name = variable.get('Label_name')
        if label_name:
            cleaned_label_name = label_name.replace("\\\\", "\\")
            cleaned_label_name = cleaned_label_name.replace("$b$-tagged", chr(92)+"btagged")
            cleaned_label_name = cleaned_label_name.replace("tau_{had}", "tauhad")
            cleaned_label_name = cleaned_label_name.replace("Higgs", chr(92)+"texttt{Higgs}")
            label_names.append(cleaned_label_name)
    return label_names

def print_latex_format(bdt_presence):
    for name, presence in bdt_presence.items():
        if presence['isInbdt_tHq_OS'] == True: x_isInbdt_tHq_OS = "x"
        elif presence['isInbdt_tHq_OS'] == False: x_isInbdt_tHq_OS = "-"
        if presence['isInbdt_ttbar_OS'] == True: x_isInbdt_ttbar_OS = "x"
        elif presence['isInbdt_ttbar_OS'] == False: x_isInbdt_ttbar_OS = "-"
        if presence['isInbdt_tHq_SS'] == True: x_isInbdt_tHq_SS = "x"
        elif presence['isInbdt_tHq_SS'] == False: x_isInbdt_tHq_SS = "-"

        print(str(name)+" & "+str(x_isInbdt_tHq_OS)+" & "+str(x_isInbdt_ttbar_OS)+" & "+str(x_isInbdt_tHq_SS)+" "+chr(92)+chr(92))



if __name__ == "__main__":
    main()



