
# coding: utf-8

# In[1]:


import pandas
import numpy as np


# In[2]:

data_file_name = '/home/sik/Documents/code/random_thoughts/sulcal_neuroanatomy/all_anterior_inferior_probabilities_RL_only_common_subjects.hdf'

All_probabilities = pandas.read_hdf(data_file_name)
All_probabilities = All_probabilities.rename({'inferior':'superior'})
All_probabilities


# In[3]:


hemispheres = list(All_probabilities.index.levels[0])
subject_ids = list(All_probabilities.index.levels[3])
sulci_names = All_probabilities.columns


# In[4]:


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




# In[5]:


all_subs_and_hems = making_tables_of_each_sub_and_hem(All_probabilities, subject_ids, hemispheres)


# In[6]:


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


# In[7]:


All_subject_and_hemisphere_sets = making_sets_of_relative_sulci_per_hemisphere_and_subject(all_subs_and_hems, sulci_names)


# ### Making sulci formulae

# In[9]:


for a in range(len(subject_ids)):
        subject = subject_ids[a]


# In[10]:


formula_for_prcs = (All_subject_and_hemisphere_sets[subject]['left']["S_central"]['anterior']).intersection(
    All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-ant-Horizont"]['superior'].union(
        All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-ant-Vertical"]['superior'].union(
            All_subject_and_hemisphere_sets[subject]['left']["Lat_Fis-post"]['superior']
        )
    )
)


# In[11]:


formula_for_sfs = ((All_subject_and_hemisphere_sets[subject]['left']["S_precentral-sup-part"]['anterior']).union(
    All_subject_and_hemisphere_sets[subject]['left']["S_precentral-inf-part"]['anterior'])).intersection(
    All_subject_and_hemisphere_sets[subject]['left']["S_front_inf"]['superior']
)


# In[12]:


formula_for_calcarine = (All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['inferior'].intersection(
    All_subject_and_hemisphere_sets[subject]['left']['S_oc_middle_and_Lunatus']['anterior']).intersection(
        All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['posterior']).intersection(
            All_subject_and_hemisphere_sets[subject]['left']['S_occipital_ant']['superior']))


# In[13]:


formula_for_pocs = ((All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['anterior']).intersection((
    All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Horizont']['superior'].union(
        All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Vertical']['superior'].union(
            All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['superior']))).intersection(
                All_subject_and_hemisphere_sets[subject]['left']['S_central']['posterior'])))


# In[14]:


formula_for_ifs = All_subject_and_hemisphere_sets[subject]['left']['S_front_sup']['inferior'].intersection(
    All_subject_and_hemisphere_sets[subject]['left']['S_precentral-sup-part']['anterior'])


# In[15]:


formula_for_pericallosal = All_subject_and_hemisphere_sets[subject]['left']['S_central']['inferior'].intersection(
    All_subject_and_hemisphere_sets[subject]['left']['S_cingul-Marginalis']['inferior']).intersection(
        All_subject_and_hemisphere_sets[subject]['left']['S_calcarine']['anterior']).intersection(
            All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-ant-Horizont']['posterior']).intersection(
                All_subject_and_hemisphere_sets[subject]['left']['S_circular_insula_inf']['superior'])


# In[16]:


formula_for_los = All_subject_and_hemisphere_sets[subject]['left']['S_parieto_occipital']['inferior'].intersection(
    All_subject_and_hemisphere_sets[subject]['left']['Lat_Fis-post']['inferior']).intersection(
       All_subject_and_hemisphere_sets[subject]['left']['S_temporal_transverse']['posterior'])


# In[17]:


formula_for_sos = All_subject_and_hemisphere_sets[subject]['left']['S_intrapariet_and_P_trans']['posterior'].intersection(
    All_subject_and_hemisphere_sets[subject]['left']['S_oc_middle_and_Lunatus']['superior'])


# # Hard-Coding Sulcal Probabilities

# ### Code for success of finding sulcus

# In[18]:


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


# In[59]:


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

# In[19]:


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


# In[20]:


PrCS_check_left = prcs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
PrCS_check_right = prcs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[21]:


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


# In[22]:


PrCS_prob_left = calculating_prob_of_PrCS_found(subject_ids, PrCS_check_left, 'Precentral_sulcus')
PrCS_prob_right = calculating_prob_of_PrCS_found(subject_ids, PrCS_check_right, 'Precentral_sulcus')


# In[27]:


print(PrCS_prob_left, PrCS_prob_right)


# In[57]:


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


# In[58]:


PrCS_left_accuracy = PrCS_formula_accuracy_check(subject_ids, PrCS_check_left)
PrCS_right_accuracy = PrCS_formula_accuracy_check(subject_ids, PrCS_check_right)


# In[65]:


PrCS_left_accuracy


# In[64]:


PrCS_right_accuracy


# ### Superior Frontal Sulcus

# In[28]:


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


# In[29]:


SFS_check_left = sfs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
SFS_check_right = sfs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[30]:


SFS_prob_left = calculating_prob_of_sulcus_found(subject_ids, SFS_check_left, 'Superior_frontal_sulcus', 'S_front_sup')
SFS_prob_right = calculating_prob_of_sulcus_found(subject_ids, SFS_check_right, 'Superior_frontal_sulcus', 'S_front_sup')


# In[31]:


print(SFS_prob_left, SFS_prob_right)


# In[61]:


SFS_left_accuracy = accuracy_check(subject_ids, SFS_check_left, 'S_front_sup')
SFS_right_accuracy = accuracy_check(subject_ids, SFS_check_right, 'S_front_sup')


# In[62]:


SFS_left_accuracy


# In[66]:


SFS_right_accuracy


# ### Calcarine Sulcus

# In[32]:


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


# In[33]:


Calcarine_check_left = calcarine_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
Calcarine_check_right = calcarine_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[34]:


Calcarine_prob_left = calculating_prob_of_sulcus_found(subject_ids, Calcarine_check_left, 'Calcarine_sulcus', 'S_calcarine')
Calcarine_prob_right = calculating_prob_of_sulcus_found(subject_ids, Calcarine_check_right, 'Calcarine_sulcus', 'S_calcarine')


# In[35]:


print(Calcarine_prob_left, Calcarine_prob_right)


# In[67]:


Calcarine_left_accuracy = accuracy_check(subject_ids, Calcarine_check_left, 'S_calcarine')
Calcarine_right_accuracy = accuracy_check(subject_ids, Calcarine_check_right, 'S_calcarine')


# In[68]:


Calcarine_left_accuracy


# In[69]:


Calcarine_right_accuracy


# ### Postcentral Sulcus

# In[36]:


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


# In[37]:


Postcentral_check_left = postcentral_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
Postcentral_check_right = postcentral_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[38]:


Postcentral_prob_left = calculating_prob_of_sulcus_found(subject_ids, Postcentral_check_left, 'Postcentral_sulcus', 'S_postcentral')
Postcentral_prob_right = calculating_prob_of_sulcus_found(subject_ids, Postcentral_check_right, 'Postcentral_sulcus', 'S_postcentral')


# In[39]:


print(Postcentral_prob_left, Postcentral_prob_right)


# In[70]:


PoCS_left_accuracy = accuracy_check(subject_ids, Postcentral_check_left, 'S_postcentral')
PoCS_right_accuracy = accuracy_check(subject_ids, Postcentral_check_right, 'S_postcentral')


# In[71]:


PoCS_left_accuracy


# In[72]:


PoCS_right_accuracy


# ### Inferior Frontal Sulcus

# In[40]:


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


# In[41]:


IFS_check_left = ifs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
IFS_check_right = ifs_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[42]:


IFS_prob_left = calculating_prob_of_sulcus_found(subject_ids, IFS_check_left, 'Inferior_frontal_sulcus', 'S_front_inf')
IFS_prob_right = calculating_prob_of_sulcus_found(subject_ids, IFS_check_right, 'Inferior_frontal_sulcus', 'S_front_inf')


# In[43]:


print(IFS_prob_left, IFS_prob_right)


# In[73]:


IFS_left_accuracy = accuracy_check(subject_ids, IFS_check_left, 'S_front_inf')
IFS_right_accuracy = accuracy_check(subject_ids, IFS_check_right, 'S_front_inf')


# In[74]:


IFS_left_accuracy


# In[75]:


IFS_right_accuracy


# ### Pericallosal Sulcus

# In[44]:


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


# In[45]:


Pericallosal_check_left = pericallosal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
Pericallosal_check_right = pericallosal_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[46]:


Pericallosal_prob_left = calculating_prob_of_sulcus_found(subject_ids, Pericallosal_check_left, 'Pericallosal_sulcus', 'S_pericallosal')
Pericallosal_prob_right = calculating_prob_of_sulcus_found(subject_ids, Pericallosal_check_right, 'Pericallosal_sulcus', 'S_pericallosal')


# In[47]:


print(Pericallosal_prob_left, Pericallosal_prob_right)


# In[76]:


Pericallosal_left_accuracy = accuracy_check(subject_ids, Pericallosal_check_left, 'S_pericallosal')
Pericallosal_right_accuracy = accuracy_check(subject_ids, Pericallosal_check_right, 'S_pericallosal')


# In[77]:


Pericallosal_left_accuracy


# In[78]:


Pericallosal_right_accuracy


# ### Lateral Occipit(o-Temporal) Sulcus

# In[48]:


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


# In[49]:


LOS_check_left = los_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
LOS_check_right = los_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[50]:


LOS_prob_left = calculating_prob_of_sulcus_found(subject_ids, LOS_check_left, 'Lateral_occipito-temporal_sulcus', 'S_oc-temp_lat')
LOS_prob_right = calculating_prob_of_sulcus_found(subject_ids, LOS_check_right, 'Lateral_occipito-temporal_sulcus', 'S_oc-temp_lat')


# In[51]:


print(LOS_prob_left, LOS_prob_right)


# In[79]:


LOS_left_accuracy = accuracy_check(subject_ids, LOS_check_left, 'S_oc-temp_lat')
LOS_right_accuracy = accuracy_check(subject_ids, LOS_check_right, 'S_oc-temp_lat')


# In[80]:


LOS_left_accuracy


# In[81]:


LOS_right_accuracy


# ### Superior Occipital Sulcus

# In[52]:


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


# In[53]:


SOS_check_left = sos_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'left')
SOS_check_right = sos_prob_from_formula(subject_ids, All_subject_and_hemisphere_sets, 'right')


# In[54]:


SOS_prob_left = calculating_prob_of_sulcus_found(subject_ids, SOS_check_left, 'Superior_occipital_sulcus', 'S_oc_sup_and_transversal')
SOS_prob_right = calculating_prob_of_sulcus_found(subject_ids, SOS_check_right, 'Superior_occipital_sulcus', 'S_oc_sup_and_transversal')


# In[55]:


print(SOS_prob_left, SOS_prob_right)


# In[82]:


SOS_left_accuracy = accuracy_check(subject_ids, SOS_check_left, 'S_oc_sup_and_transversal')
SOS_right_accuracy = accuracy_check(subject_ids, SOS_check_right, 'S_oc_sup_and_transversal')


# In[83]:


SOS_left_accuracy


# In[84]:


SOS_right_accuracy
