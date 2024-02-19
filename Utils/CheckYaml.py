## Reads the configuration file of mva_runner to check if there is any repeated element. Repeated elements (indexes or variables) cause mva runner to stop
import yaml
from itertools import chain


def main():
    config_path = "/lhome/ific/p/pamarag/Work/New_BDT/tHqIFIC/tHqMVA/Utils/config_SS.yaml"
    #variables_set = 'ShortMVAttbar'
    variables_set = 'ShortMVAtH'
    #variables_set = 'ShortMVAttbar' 
    #variables_set = 'GeneralMVA'
#    config_content = OpenYaml(config_path, variables_set)

    yamlfile = open(config_path, 'r')

    config = yaml.load(yamlfile, Loader=yaml.BaseLoader)
    config_content = config[variables_set]

    list_of_variables= []
    list_of_keys = []
    for key, value in config_content.items():
        list_of_keys.append(str(key))
        aux = value
        list_of_variables.append(aux['Name'])
        if False:
            try: 
                #print("" + str(aux['Label_name']))
                print(r'\begin{frame}{Discriminant variables ::  '+str(aux['Label_name'])+'}')
                print(r'   \begin{figure}')
                print(r'   \centering')
                print(r'    \includegraphics[width = 0.45\textwidth]{DiscriminantDistributions/PR_'+str(aux['Name'])+'.pdf}')
                print(r'    \includegraphics[width = 0.45\textwidth]{DiscriminantDistributions/stack/PR_'+str(aux['Name'])+'.pdf}')
                #print("    \caption*{"+str(aux['Label_name'])+"}")
                print("\end{figure}")
                print("\end{frame}")

            except: print(aux['Name'])
       # print("\item " + aux['Label_name'])
    #print "list_of_keys = " + str(list_of_keys)
    #print "list_of_variables = " + str(list_of_variables)
    print "len(list_of_keys) = " + str(len(list_of_keys))
    print "len(list_of_variables) = " + str(len(list_of_variables))
    #exit()
    if finduplicate(list_of_keys) == False: print "No duplicated keys in " + str(variables_set)
    if finduplicate(list_of_variables) == False: print "No duplicated variables in " + str(variables_set)

def OpenYaml(path, variable_set, option = 'r'):
    with open(path, option) as stream:
        out = yaml.safe_load(stream) # <- This is a dictionary so it is not able to have duplicated keys
    return out[variable_set]

def finduplicate(thelist):
    l1 = []
    AreDuplicated = False
    for i in thelist:

        if i in l1:
            print("Duplicated key/variable: "+ i)
            AreDuplicated = True
        else:
            l1.append(i) 

    return AreDuplicated

#def ifducplicates(thelist):
#    if len(thelist) != len(set(thelist)):
#        print "There are duplicates"
#        return True
#    else:
#        return False

if __name__ == '__main__':
  main()
