# coding: utf-8
import pandas
import numpy as np

#data_file_name = '/home/sik/Documents/code/random_thoughts/sulcal_neuroanatomy/all_anterior_inferior_probabilities_RL_only_common_subjects.hdf'
#data_file_name = '/home/amachlou/Documents/INRIA/Sulcal_Neuroanatomy/all_anterior_inferior_probabilities_RL_only_common_subjects.hdf'

def get_probabilities_hemispheres_ids_sulcinames(data_file_name):
    All_probabilities = pandas.read_hdf(data_file_name)
    All_probabilities = All_probabilities.rename({'inferior': 'superior'})

    hemispheres = list(All_probabilities.index.levels[0])
    subject_ids = list(All_probabilities.index.levels[3])
    sulci_names = All_probabilities.columns

    return All_probabilities, hemispheres, subject_ids, sulci_names


def making_tables_of_each_sub_and_hem(all_probs, subject_ids, hemispheres):
    '''
    Input: pandas dataframe table of all subjects sulcal relations, both hemispheres
    Retrieves information of each subject's and each hemisphere's probabilities, and
    creates a dataframe table per subject per hemisphere
    '''
    subjects = {}

    for subject in subject_ids:
        hemisphere = {}
        for hem in hemispheres:
            sub_and_hem = all_probs.loc[(hem, slice(None), slice(None), subject)]
            hemisphere[hem]= sub_and_hem
        subjects[subject]=hemisphere

    return subjects


def making_sets_of_relative_sulci_per_hemisphere_and_subject(all_subs_and_hems, sulci_names):
    '''
    Creates a quadruple dictionary with keys: subject number, hemisphere, reference sulcus, relative position

    all_sulci_sets[subject][hemisphere][reference_sulcus][relative_position]

    Parameters: all_subs_and_hems: dictionary with keys: subject number, hemisphere
                sulci_names: pandas dataframe index array
                             names of sulci in one hemisphere
    Returns: all_sulci_sets: dict

    '''

    all_sulci_sets = {}

    for subject in all_subs_and_hems:

        hemisphere = {}

        for hem in all_subs_and_hems[subject]:

            sulci_sets = {}

            for sulcus_name in sulci_names:

                current_sub_and_hem = all_subs_and_hems[subject][hem]

                anterior_probability = current_sub_and_hem.xs('anterior', level='probability_of')
                superior_probability = current_sub_and_hem.xs('superior', level='probability_of')
                anterior_sulci=set(anterior_probability[sulcus_name][(anterior_probability>0.95)[sulcus_name]].index)
                posterior_sulci=set(anterior_probability[sulcus_name][(anterior_probability<0.05)[sulcus_name]].index)
                superior_sulci=set(superior_probability[sulcus_name][(superior_probability>0.95)[sulcus_name]].index)
                inferior_sulci=set(superior_probability[sulcus_name][(superior_probability<0.05)[sulcus_name]].index)
                position = {}
                position['anterior']=anterior_sulci
                position['posterior']=posterior_sulci
                position['superior']=superior_sulci
                position['inferior']=inferior_sulci
                sulci_sets[sulcus_name]=position

            hemisphere[hem]=sulci_sets

        all_sulci_sets[subject]=hemisphere

    return all_sulci_sets

# ### Code for success of finding sulcus
def calculating_prob_of_sulcus_found(subject_ids, sulci_output, desired_sulcus, destrieux_name):
    '''
    Input: subject_ids=list of subject ids,
            sulci_output=dictionary, key: subject id, value: names of sulci returned from formula to find one sulcus
    Function iterates through sulci_output dictionary to check if desired sulcus is found in subject or not.
    Output: Total number of subjects for which desired sulcus was found
            Total number of subjects
            Probability of formula returning the desired sulcus within a subject.
    '''

    desired_sulcus = []

    for current_subject in subject_ids:
        check_for_sulcus = destrieux_name in sulci_output[current_subject]
        desired_sulcus.append(check_for_sulcus)

    prob_of_sulcus_found = (sum(desired_sulcus))/(len(desired_sulcus))

    return sum(desired_sulcus), len(desired_sulcus), prob_of_sulcus_found

def accuracy_check(subject_ids, sulcus_check, destrieux_name):
    '''
    Input: dictionary with subject id as key, list of found sulci from sulcal formula as values
    Function searches for desired sulcus within values, and if found,
    Returns: dictionary of subject for which search was successful, with length of list of sulci found
            : Number of subjects for which only desired sulcus was found
            : Proportion of accuracy
    '''

    sensitivity_output = {}
    accuracy_of_output = 0

    for current_subject in subject_ids:
        if destrieux_name in sulcus_check[current_subject]:
            sensitivity_output[current_subject]=len(sulcus_check[current_subject])
            if len(sulcus_check[current_subject]) is 1:
                accuracy_of_output +=1

    return  sensitivity_output, accuracy_of_output, accuracy_of_output / (len(sensitivity_output))


# ### Precentral Sulcus
def prcs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = (current_subject_hemi["S_central"]['anterior']).intersection(
        current_subject_hemi["Lat_Fis-ant-Horizont"]['superior'].union(
            current_subject_hemi["Lat_Fis-ant-Vertical"]['superior'].union(
                current_subject_hemi["Lat_Fis-post"]['superior']
                )
            )
        )

        sulci_found[current_subject]=sulcus_test

    return sulci_found

def calculating_prob_of_PrCS_found(subject_ids, sulci_output, desired_sulcus):
    '''
    Input: subject_ids=list of subject ids,
            sulci_output=dictionary, key: subject id, value: names of sulci returned from formula to find one sulcus
    Function iterates through sulci_output dictionary to check if desired sulcus is found in subject or not.
    Output: Total number of subjects for which desired sulcus was found
            Total number of subjects
            Probability of formula returning the desired sulcus within a subject.
    '''

    desired_sulcus = []

    for current_subject in subject_ids:
        check_for_sulcus = 'S_precentral-inf-part' in sulci_output[current_subject] and 'S_precentral-sup-part' in sulci_output[current_subject]
        desired_sulcus.append(check_for_sulcus)

    prob_of_sulcus_found = (sum(desired_sulcus))/(len(desired_sulcus))

    return sum(desired_sulcus), len(desired_sulcus), prob_of_sulcus_found


def PrCS_formula_accuracy_check(subject_ids, sulcus_check):
    '''
    Input: dictionary with subject id as key, list of found sulci from sulcal formula as values
    Function searches for desired sulcus within values, and if found,
    Returns: dictionary of subject for which search was successful, with length of list of sulci found
            : Number of subjects for which only desired sulcus was found
            : Proportion of accuracy
    '''

    sensitivity_output = {}
    accuracy_of_output = 0

    for current_subject in subject_ids:
        if 'S_precentral-sup-part' in sulcus_check[current_subject] and 'S_precentral-inf-part' in sulcus_check[current_subject]:
            sensitivity_output[current_subject]=len(sulcus_check[current_subject])
            if len(sulcus_check[current_subject]) is 1:
                accuracy_of_output +=1

    return  sensitivity_output, accuracy_of_output, accuracy_of_output / (len(sensitivity_output))


def marginal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_pericallosal']['superior'].intersection(
            current_subject_hemi['S_postcentral']['anterior']).intersection(
                current_subject_hemi['S_central']['posterior']
        )

        sulci_found[current_subject] = sulcus_test

    return sulci_found


def ips_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_central']['posterior'].intersection(
            current_subject_hemi['S_parieto_occipital']['anterior']).intersection(
            current_subject_hemi['S_interm_prim-Jensen']['superior']).intersection(
            current_subject_hemi['S_front_sup']['inferior']
        )

        sulci_found[current_subject] = sulcus_test

    return sulci_found


def coll_transv_post_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_temporal_inf']['inferior']

        sulci_found[current_subject] = sulcus_test

    return sulci_found


def coll_transv_ant_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_oc-temp_lat']['inferior']

        sulci_found[current_subject] = sulcus_test

    return sulci_found


def fms_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = (current_subject_hemi['S_front_sup']['inferior']).intersection(
            current_subject_hemi['Lat_Fis-ant-Horizont']['superior']).intersection(
            current_subject_hemi['S_precentral-inf-part']['anterior'])
        sulci_found[current_subject] = sulcus_test

    return sulci_found


def tts_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = (current_subject_hemi['Lat_Fis-post']['inferior']).intersection(
            current_subject_hemi['Lat_Fis-ant-Horizont']['posterior'])
        sulci_found[current_subject] = sulcus_test

    return sulci_found


def its_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = (current_subject_hemi['S_temporal_sup']['inferior']).intersection(
            current_subject_hemi['Lat_Fis-ant-Horizont']['posterior']).intersection(
            current_subject_hemi['S_oc-temp_lat']['anterior']).intersection(
            current_subject_hemi['S_collat_transv_ant']['superior'])
        sulci_found[current_subject] = sulcus_test

    return sulci_found


def sts_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_intrapariet_and_P_trans']['inferior'].intersection(
                current_subject_hemi['S_oc_sup_and_transversal']['anterior']).intersection(
                    current_subject_hemi['S_front_middle']['posterior']).intersection(
                        current_subject_hemi['S_temporal_inf']['superior']
        )


        sulci_found[current_subject] = sulcus_test

    return sulci_found

def subparietal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_parieto_occipital']['superior'].intersection(
   current_subject_hemi['S_postcentral']['posterior']).intersection(
   current_subject_hemi['S_cingul-Marginalis']['inferior'])
        sulci_found[current_subject]=sulcus_test

    return sulci_found

# ### Superior Frontal Sulcus
def sfs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = ((current_subject_hemi["S_precentral-sup-part"]['anterior']).union(
            current_subject_hemi["S_precentral-inf-part"]['anterior'])).intersection(
            current_subject_hemi["S_front_inf"]['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# ### Calcarine Sulcus
def calcarine_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_parieto_occipital']['inferior'].intersection(
                current_subject_hemi['Lat_Fis-post']['posterior']).intersection(
                    current_subject_hemi['S_occipital_ant']['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# ### Postcentral Sulcus
def postcentral_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = (current_subject_hemi['S_interm_prim-Jensen']['anterior']).intersection((
            current_subject_hemi['S_temporal_sup']['superior']).intersection(
                       current_subject_hemi['S_central']['posterior']))

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# ### Inferior Frontal Sulcus
def ifs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_front_sup']['inferior'].intersection(
           current_subject_hemi['S_precentral-sup-part']['anterior']).intersection(
                current_subject_hemi['S_orbital_lateral']['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# ### Pericallosal Sulcus
def pericallosal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_cingul-Marginalis']['inferior'].intersection(
            current_subject_hemi['S_occipital_ant']['anterior']).intersection(
                current_subject_hemi['S_oc-temp_lat']['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# ### Lateral Occipit(o-Temporal) Sulcus
def los_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_parieto_occipital']['inferior'].intersection(
            current_subject_hemi['Lat_Fis-post']['posterior']).intersection(
               current_subject_hemi['S_oc_middle_and_Lunatus']['anterior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found

def suborbital_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_temporal_inf']['anterior'].intersection(
           current_subject_hemi['S_front_inf']['inferior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found

def interm_jen_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_intrapariet_and_P_trans']['inferior'].intersection(
           current_subject_hemi['S_postcentral']['posterior']).intersection(
               current_subject_hemi['S_oc_sup_and_transversal']['anterior'])


        sulci_found[current_subject]=sulcus_test

    return sulci_found

# ### Occipital Middle and Lunatus Sulcus
def oc_mid_lun_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_temporal_sup']['posterior'].intersection(
           current_subject_hemi['S_subparietal']['inferior']).intersection(
               current_subject_hemi['S_suborbital']['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found

# ### Superior Occipital Sulcus
def sos_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, hemi):
    '''
    Input: subject_ids=list of all subjects' ids, used as keys in All_subject_and_hemisphere_sets=quadruple dictionary
    in pandas dataframe format, of subject, hemisphere, sulcus, set of sulci in relative position
    Function calculates relationship of how to find PrCS for each subject's left hemisphere
    Output: dictionary with subject id as key, and list of sulci as values.
    '''

    sulci_found = {}

    for current_subject in subject_ids:
        current_subject_hemi = All_subject_and_hemisphere_sets[current_subject][hemi]
        sulcus_test = current_subject_hemi['S_intrapariet_and_P_trans']['posterior'].intersection(
           current_subject_hemi['S_oc_middle_and_Lunatus']['superior']).intersection(
                current_subject_hemi['S_precentral-sup-part']['inferior']
        )

        sulci_found[current_subject]=sulcus_test

    return sulci_found


# SFS_check_left = sfs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# SFS_check_right = sfs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# SFS_prob_left = calculating_prob_of_sulcus_found(subject_ids, SFS_check_left, 'Superior_frontal_sulcus', 'S_front_sup')
# SFS_prob_right = calculating_prob_of_sulcus_found(subject_ids, SFS_check_right, 'Superior_frontal_sulcus', 'S_front_sup')

# SFS_left_accuracy = accuracy_check(subject_ids, SFS_check_left, 'S_front_sup')
# SFS_right_accuracy = accuracy_check(subject_ids, SFS_check_right, 'S_front_sup')

# SFS_left_accuracy

# SFS_right_accuracy




# Calcarine_check_left = calcarine_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# Calcarine_check_right = calcarine_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# Calcarine_prob_left = calculating_prob_of_sulcus_found(subject_ids, Calcarine_check_left, 'Calcarine_sulcus', 'S_calcarine')
# Calcarine_prob_right = calculating_prob_of_sulcus_found(subject_ids, Calcarine_check_right, 'Calcarine_sulcus', 'S_calcarine')

# Calcarine_left_accuracy = accuracy_check(subject_ids, Calcarine_check_left, 'S_calcarine')
# Calcarine_right_accuracy = accuracy_check(subject_ids, Calcarine_check_right, 'S_calcarine')

# Calcarine_left_accuracy

# Calcarine_right_accuracy



# Pericallosal_check_left = pericallosal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# Pericallosal_check_right = pericallosal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# Pericallosal_prob_left = calculating_prob_of_sulcus_found(subject_ids, Pericallosal_check_left, 'Pericallosal_sulcus', 'S_pericallosal')
# Pericallosal_prob_right = calculating_prob_of_sulcus_found(subject_ids, Pericallosal_check_right, 'Pericallosal_sulcus', 'S_pericallosal')

# Pericallosal_left_accuracy = accuracy_check(subject_ids, Pericallosal_check_left, 'S_pericallosal')
# Pericallosal_right_accuracy = accuracy_check(subject_ids, Pericallosal_check_right, 'S_pericallosal')

# Pericallosal_left_accuracy

# Pericallosal_right_accuracy



# IFS_check_left = ifs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# IFS_check_right = ifs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# IFS_prob_left = calculating_prob_of_sulcus_found(subject_ids, IFS_check_left, 'Inferior_frontal_sulcus', 'S_front_inf')
# IFS_prob_right = calculating_prob_of_sulcus_found(subject_ids, IFS_check_right, 'Inferior_frontal_sulcus', 'S_front_inf')

# IFS_left_accuracy = accuracy_check(subject_ids, IFS_check_left, 'S_front_inf')
# IFS_right_accuracy = accuracy_check(subject_ids, IFS_check_right, 'S_front_inf')

# IFS_left_accuracy

# IFS_right_accuracy


# Postcentral_check_left = postcentral_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# Postcentral_check_right = postcentral_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# Postcentral_prob_left = calculating_prob_of_sulcus_found(subject_ids, Postcentral_check_left, 'Postcentral_sulcus', 'S_postcentral')
# Postcentral_prob_right = calculating_prob_of_sulcus_found(subject_ids, Postcentral_check_right, 'Postcentral_sulcus', 'S_postcentral')

# PoCS_left_accuracy = accuracy_check(subject_ids, Postcentral_check_left, 'S_postcentral')
# PoCS_right_accuracy = accuracy_check(subject_ids, Postcentral_check_right, 'S_postcentral')

# PoCS_left_accuracy

# PoCS_right_accuracy




# LOS_check_left = los_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# LOS_check_right = los_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# LOS_prob_left = calculating_prob_of_sulcus_found(subject_ids, LOS_check_left, 'Lateral_occipito-temporal_sulcus', 'S_oc-temp_lat')
# LOS_prob_right = calculating_prob_of_sulcus_found(subject_ids, LOS_check_right, 'Lateral_occipito-temporal_sulcus', 'S_oc-temp_lat')

# LOS_left_accuracy = accuracy_check(subject_ids, LOS_check_left, 'S_oc-temp_lat')
# LOS_right_accuracy = accuracy_check(subject_ids, LOS_check_right, 'S_oc-temp_lat')

# LOS_left_accuracy

# LOS_right_accuracy




# SOS_check_left = sos_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
# SOS_check_right = sos_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')

# SOS_prob_left = calculating_prob_of_sulcus_found(subject_ids, SOS_check_left, 'Superior_occipital_sulcus', 'S_oc_sup_and_transversal')
# SOS_prob_right = calculating_prob_of_sulcus_found(subject_ids, SOS_check_right, 'Superior_occipital_sulcus', 'S_oc_sup_and_transversal')

# SOS_left_accuracy = accuracy_check(subject_ids, SOS_check_left, 'S_oc_sup_and_transversal')
# SOS_right_accuracy = accuracy_check(subject_ids, SOS_check_right, 'S_oc_sup_and_transversal')

# SOS_left_accuracy

# SOS_right_accuracy




# # ### Making sulci formulae
# for a in range(len(subject_ids)):
#         subject = subject_ids[a]

# formula_for_prcs = (All_subject_and_hemisphere_sets[subject]['left']["S_central"]['anterior']).intersection(
#     All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-ant-Horizont"]['superior'].union(
#         All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-ant-Vertical"]['superior'].union(
#             All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-post"]['superior']
#         )
#     )
# )

# formula_for_sfs = ((All_subject_and_hemisphere_sets[subject]['left']["S_precentral-sup-part"]['anterior']).union(
#     All_subject_and_hemisphere_sets[subject]['left']["S_precentral-inf-part"]['anterior'])).intersection(
#     All_subject_and_hemisphere_sets[subject]['left']["S_front_inf"]['superior']
# )

# formula_for_calcarine = (All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['inferior'].intersection(
#     All_subject_and_hemisphere_sets[subject]['left']['S_oc_middle_and_Lunatus']['anterior']).intersection(
#         All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['posterior']).intersection(
#             All_subject_and_hemisphere_sets[subject]['left']['S_occipital_ant']['superior']))

# formula_for_pocs = ((All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['anterior']).intersection((
#     All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Horizont']['superior'].union(
#         All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Vertical']['superior'].union(
#             All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['superior']))).intersection(
#                 All_subject_and_hemisphere_sets[subject]['left']['S_central']['posterior'])))

# formula_for_ifs = All_subject_and_hemisphere_sets[subject]['left']['S_front_sup']['inferior'].intersection(
#     All_subject_and_hemisphere_sets[subject]['left']['S_precentral-sup-part']['anterior'])

# formula_for_pericallosal = All_subject_and_hemisphere_sets[subject]['left']['S_central']['inferior'].intersection(
#     All_subject_and_hemisphere_sets[subject]['left']['S_cingul-Marginalis']['inferior']).intersection(
#         All_subject_and_hemisphere_sets[subject]['left']['S_calcarine']['anterior']).intersection(
#             All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Horizont']['posterior']).intersection(
#                 All_subject_and_hemisphere_sets[subject]['left']['S_circular_insula_inf']['superior'])

# formula_for_los = All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['inferior'].intersection(
#     All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['inferior']).intersection(
#        All_subject_and_hemisphere_sets[subject]['left']['S_temporal_transverse']['posterior'])

# formula_for_sos = All_subject_and_hemisphere_sets[subject]['left']['S_intrapariet_and_P_trans']['posterior'].intersection(
#     All_subject_and_hemisphere_sets[subject]['left']['S_oc_middle_and_Lunatus']['superior'])
