# coding: utf-8
import pandas
import numpy as np


All_probabilities = pandas.read_hdf(data_file_name)
All_probabilities = All_probabilities.rename({'inferior':'superior'})

hemispheres = list(All_probabilities.index.levels[0])
subject_ids = list(All_probabilities.index.levels[3])
sulci_names = All_probabilities.columns

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
        sulcus_test = (current_subject_hemi['S_parieto_occipital']['inferior'].intersection(
            current_subject_hemi['S_oc_middle_and_Lunatus']['anterior']).intersection(
                current_subject_hemi['Lat_Fis-post']['posterior']).intersection(
                    current_subject_hemi['S_occipital_ant']['superior']))

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
        sulcus_test = (current_subject_hemi['S_parieto_occipital']['anterior']).intersection((
            current_subject_hemi['Lat_Fis-ant-Horizont']['superior'].union(
               current_subject_hemi['Lat_Fis-ant-Vertical']['superior'].union(
                  current_subject_hemi['Lat_Fis-post']['superior']))).intersection(
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
           current_subject_hemi['S_precentral-sup-part']['anterior'])

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
        sulcus_test = current_subject_hemi['S_central']['inferior'].intersection(
            current_subject_hemi['S_cingul-Marginalis']['inferior']).intersection(
                current_subject_hemi['S_calcarine']['anterior']).intersection(
                   current_subject_hemi['Lat_Fis-ant-Horizont']['posterior']).intersection(
                        current_subject_hemi['S_circular_insula_inf']['superior'])

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
            current_subject_hemi['Lat_Fis-post']['inferior']).intersection(
               current_subject_hemi['S_temporal_transverse']['posterior'])

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
           current_subject_hemi['S_oc_middle_and_Lunatus']['superior'])

        sulci_found[current_subject]=sulcus_test

    return sulci_found

