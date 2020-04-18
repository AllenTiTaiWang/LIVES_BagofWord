import pandas as pd
import xml.etree.ElementTree as ET
import glob
import json
import sys
import os
cwd = os.getcwd()



def read_tables(label):
    """
    This function reads the fidelity measures table, and drops the 
    rows (transcripts) with NA value in certain fidelity measure (label).
    It also prints the count of each value in the label.

    :param label: A string stands for one of fidelity measure.
    ex: availability.
    :return: A pandas dataframe with no NA value in a specific measure.
    """
    #Read the table 
    RES_score = pd.read_excel(os.path.join(cwd, '../Full dataset of fidelity scored calls with outcome data/LIVES_Call-Level Data_all fidelity scored calls n=323.xls'))
    table = RES_score
    #Drop NA rows
    table = table.dropna(axis=0, subset = [label])
    #Shows the count of each value
    print(table[label].value_counts())
    return table

def read_tables_subtract(label1, label2):
    """
    This function reads both behavior outcomes table and fidelity measures table,
    and then union tham on `sid` column. Also, it drops the transcripts which has
    `call_number` larger than 17. Eventually, it talkes label1 value to subtract
    label2 value (ex: tfat_pcal_1 and tfat_pcal_2) and save it under `difference`
    column, and only keep the most recent call if the participant has multiple calls.

    :param label1: A string stands for one of the behavior outcome (ex: tfat_pcal_1).
    :param label2: A string matches the label1 (ex: tfat_pcal_2)
    :return pandas dataframe with a `difference` column in it:
    """
    #Read the tables
    RES_score = pd.read_excel(os.path.join(cwd, '../Full dataset of fidelity scored calls with outcome data/LIVES_Call-Level Data_all fidelity scored calls n=323.xls'))
    RES_outcome = pd.read_excel(os.path.join(cwd, '../Full dataset of fidelity scored calls with outcome data/LIVES_Outcome Data_BL-24M_ppts from fidelity list.xlsx'))
    #Merge on SID
    table = pd.merge(RES_score, RES_outcome, on='sid', how='outer')
    #Delete the calls after 6 months
    indexnames = table[table['call_number'] > 17].index
    table.drop(indexnames, inplace=True)
    #Drop NA
    table = table.dropna(axis=0, subset = [label1, label2, 'call_number'])
    table['difference'] = table[label2] - table[label1]
    #Keep the most recent cal
    table = table.sort_values(by='call_number', ascending=False)
    table = table.drop_duplicates(subset=['sid'])
    return table


def split_url(url):
    """
    The function splits an url, and only keep audio id.

    :param url: A string of url
    :return: Audio ID
    """
    url_final = url.split("/")[-1].strip()
    return url_final

def text_to_dic(people, table, label):
    """
    This function reads all transcripts under file name `TRS files n=323`. It splits
    the sentences by different speaker for next step. each call would be saved in a 
    tuple with its matching label as a value in a dictionary called url which has keys
    of url.

    :param people: A string could be 'both', 'coach', or 'participant' used to seperate
    the sentences.
    :param table: The dataframe from `read_table` or `read_table_subtract`.
    :param label: A tring stands for fidelity measure or `difference`
    :return: A dictionary includes url as key and a tuple with conversation and label
    """
    #Read transcipts
    url_dic = {}
    count = 0
    for transcript in glob.glob(os.path.join(cwd, "../Full dataset of fidelity scored calls with outcome data/TRS files n=323/*.trs")):
        #Get audio ID from its title
        url = transcript.split("/")[-1].split(" ")[-1].split(".")[0]
        #Parse the transcript
        xml = ET.parse(transcript)
        conversation = ""
        #Save the sentence depends on the parameter people
        if "both" in people:
            for turn in xml.iter('Turn'):
                word = ''.join(turn.itertext())
                conversation += word + ' '

        elif "participant" in people or "coach" in people:
            for spk in xml.iter('Speaker'):
                if spk.attrib['name'] == people:
                    target_spkr = spk.attrib['id']

            for turn in xml.iter('Turn'):
                if turn.attrib['speaker'] == target_spkr:
                    word = ''.join(turn.itertext())
                    conversation += word + ' ' 
        #Keep the data in dictionary format
        if sum(table["call_url"].str.contains(url)) > 0:
            label_t = table.loc[table["call_url"].apply(split_url) == url, label].values[0]
            url_dic[url] = (conversation, float(label_t))
            count += 1
    
    return url_dic

