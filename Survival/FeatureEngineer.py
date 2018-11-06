import pandas as pd
import numpy as np
from scipy import stats
import math
import re
from collections import defaultdict
from collections import Counter


class FeatureEngineer():

    def __init__(self, verbose=True, data_path=None):
        if data_path is None:
            self.DATA_PATH = \
                "../../../../media/latte/npsurvival/preprocessed/" + \
                "{SOURCE}_forecast_icu_los/{FILENAME}"
        else:
            self.DATA_PATH = data_path + "{SOURCE}_forecast_icu_los/{FILENAME}"
        self.SOURCES = ["pancreatitis", "ich", "sepsis"]
        self.verbose = verbose


    def get_file_path(self, source_index, filename):
        return self.DATA_PATH.format(SOURCE=self.SOURCES[source_index],
            FILENAME=filename)


    def value_data_cleaning(self, piece_value, piece_event):

        na_set = {"", "error", "notdone", "none", "**_info_not_available_**",
                  'unable_t_report', "unable_to_report",
                  'error,unable_to_report', "discarded",
                  'computer_network_failure._test_not_resulted.',
                  "error_specimen_clotted", "specimen_clotted",
                  'disregard_result', "test_not_done", 'unable_to_determine',
                  'unable_to_determine:', "unknown", 'unable_to_repoet',
                  'unable_to_repert', 'not_reported', 'specimen_clottted',
                  'not_done', "unable", "unable_to_repot", 'spec._clotted',
                  'spec.clotted', 'unable_to_analyze'}
       
        dup_value_ref = {
            "chart:marital_status": {'w': 'widowed', 'm': 'married',
                's': 'single', 'd': 'divored'},
            "chart:marital_status:na": {'w': 'widowed', 'm': 'married',
                's': 'single', 'd': 'divored'},
            "lab:urine:hematology:bacteria": {'o': 'occ', 'm': 'mod',
                'f': 'few'},
            "lab:urine:hematology:glucose": {'tr': 'na', 'neg': -1},
            "lab:urine:hematology:ketone": {'tr': 'na', 'neg': -1},
            "lab:urine:hematology:protein": {'tr': 'na', 'neg': -1},
            "lab:urine:hematology:urine_appearance": {"slcloudy": "slcldy"},
            "lab:urine:hematology:urine_color": {'y': 'yellow',
                'yel': 'yellow', 's': 'straw', 'drkamber': 'dkamber',
                'dkambe': 'dkamber', 'dkamb': 'dkamber', 'amb': 'amber'},
            "lab:urine:hematology:urobilinogen": {'neg': -1},
            "lab:blood:chemistry:ethanol": {'neg': -1},
            "lab:blood:chemistry:acetaminophen": {'neg': -1},
            "lab:blood:chemistry:salicylate": {'neg': -1}
        }
       
        zero_one_special = {
            "lab:urine:hematology:bacteria": {0: "zero"},
            "chart:parameters_checked:na": {1: 'yes'},
            "lab:blood:hematology:anisocytosis": {1: '1+'},
            "lab:blood:hematology:poikilocytosis": {1: '1+'},
            "lab:urine:hematology:yeast": {0: 'zero'}
        }

        # convert value from str into float, if possible
        try:
            piece_value = float(piece_value)
            if piece_value == 1 or piece_value == 0:
                if piece_event in zero_one_special:
                    piece_value = zero_one_special[piece_event][piece_value]
        except:
            piece_value = piece_value.replace(" ", "_")
            pattern_1 = r'.*([0-9]+)_?-_?([0-9]+).*'
            pattern_2 = r'[a-z_:]*(<|>|<=|>=|less_than|greater_than' + \
                r'|greater_thn|less_thn)_?([0-9]+).*'
            pattern_3 = r'([0-9]+)_?(cm|mm|grams|gmmag|mgso4|gm_mag|gmg+|o1b)'
            if re.match(pattern_1, piece_value):
                result = re.match(pattern_1, piece_value)
                piece_value = (float(result[1]) + float(result[2])) / 2
            elif re.match(pattern_2, piece_value):
                result = re.match(pattern_2, piece_value)
                piece_value = float(result[2])
            elif re.match(pattern_3, piece_value):
                result = re.match(pattern_3, piece_value)
                piece_value = float(result[1])

        if type(piece_value) == float and np.isnan(piece_value):
            piece_value = "na"

        if type(piece_value) == str:
            piece_value = piece_value.replace(" ", "_")
            if piece_value in na_set or \
                re.match(r'unable_to.*due_to.*', piece_value):
                piece_value = 'na'
            elif piece_event in dup_value_ref:
                if piece_value in dup_value_ref[piece_event]:
                    piece_value = dup_value_ref[piece_event][piece_value]
            else:
                pattern = r".*(disregard_previous|error_previously" + \
                    r"_reported|disregard_result).*"
                if re.match(pattern, piece_value):
                    piece_value = "na"
        return piece_value


    def scalarize_numerical_value_list(self, value_list):
    #     return np.mean(value_list)
        return value_list[-1] # use latest information


    def remove_na_in_list(self, source_list):
        return [item for item in source_list
                if not ((type(item) == float and np.isnan(item)) or \
                        (type(item) == str and item == "na") or \
                        item is None or item == "")]


    def to_float_in_list(self, source_list):
        output_list = []
        for item in source_list:
            try:
                output_list.append(float(item))
            except:
                pass
        return output_list


    def check_same_distribution(self, list1, list2):
        list1_float = self.to_float_in_list(list1)
        list2_float = self.to_float_in_list(list2)

        n_1 = len(list1_float)
        n_2 = len(list2_float)
       
        if n_1 < 30 or n_2 < 30:
            return True
       
        mu_1 = np.mean(list1_float)
        mu_2 = np.mean(list2_float)
        std_1 = np.std(list1_float)
        std_2 = np.std(list2_float)

        z_score = (mu_1 - mu_2) / np.sqrt(std_1**2 / n_1 + std_2**2 / n_2)
        p_values = stats.norm.sf(abs(z_score))*2
       
        if p_values > 0.01:
            return True
        else:
            return False


    def pounds_to_kg(self, pounds):
        if type(pounds) == str:
            return pounds
        kilograms = pounds / 2.2
        return int(kilograms)


    def process_event_ends_with_colon(self, target_event, source_df):

        # look for event that contains the target event which ends with ":"
        derive_events = []
        for event in source_df["event"].unique():
            if target_event in event and target_event != event:
                derive_events.append(event)

        # get value list of the target_event
        val_list = list(source_df[source_df["event"] == target_event]["value"])
        val_list = self.remove_na_in_list(val_list)
       
        if len(derive_events) == 0: # if no other event that contains the target
            if len(val_list) == 0: # all values are 'na' or np.nan or None or ""
                output_event = target_event + "na"
            else: # the event_name won't be split later
                output_event = target_event[:-1]
        else:
            output_event = target_event
           
            for derive_event in derive_events:
                derive_val_list = list(source_df[source_df["event"] == \
                    derive_event]["value"])
                derive_val_list = self.remove_na_in_list(derive_val_list)
           
                if len(val_list) == 0:
                    output_event = derive_event
                    val_list.extend(derive_val_list)
                elif len(derive_val_list) == 0:
                    val_list.extend(derive_val_list)
                else:
                    if self.check_same_distribution(val_list, derive_val_list):
                        if output_event in derive_event:
                            output_event = derive_event
                        val_list.extend(derive_val_list)
                    else:
                        pass

            if output_event.endswith(":"):
                if len(val_list) == 0: # all values are 'na'|np.nan|None|""
                    output_event = output_event + "na"
                else: # the event_name won't be split later
                    output_event = output_event[:-1]
       
        return output_event


    def from_df_to_dict(self, df):
       
        event_change_coded_table = {
            "chart:admit wt": "chart:admit wt:kg",
            "chart:bun": "chart:bun:mg/dl",
            "chart:creatinine": "chart:creatinine:mg/dl",
            "chart:tidal volume (observed)": \
                "chart:tidal volume (observed):ml/b",
            "chart:tidal volume (set)": "chart:tidal volume (set):ml/b"
        }

        event_change_dict = {}
        patient_dict = defaultdict(lambda: defaultdict(list))
        na_event_set = set([])
        non_na_event_set = set([])

        if self.verbose:
            print("[LOG]Process df row by row and update dict.")

        matrix = df.values
        feature_idx = {feature:idx for idx, feature in enumerate(df.columns)}
        for pieces in matrix:
            patient_id = int(float(pieces[feature_idx["patientID"]]))
            piece_event = pieces[feature_idx["event"]]
            piece_value = self.value_data_cleaning(pieces[feature_idx["value"]], piece_event)
            if piece_event.endswith(":"):
                if piece_event in event_change_dict:
                    piece_event = event_change_dict[piece_event]
                else:
                    piece_event = self.process_event_ends_with_colon(
                        piece_event, df)
                    event_change_dict[pieces[feature_idx["event"]]] = piece_event
               
            if piece_event == "chart:admit wt":
                piece_value = self.pounds_to_kg(piece_value) # lbs to kg
            if piece_event in event_change_coded_table:
                piece_event = event_change_coded_table[piece_event]
            piece_event = piece_event.replace(" ", "_")

            patient_dict[patient_id][piece_event].append(piece_value)

            if piece_value == 'na' or \
                (type(piece_value) == float and np.isnan(piece_value)):
                na_event_set.add(piece_event)
            else:
                non_na_event_set.add(piece_event)

        all_na_event_set = na_event_set - non_na_event_set
        return patient_dict, all_na_event_set


    def read_text_file(self, src_idx, prefix, tag, suffix):
        fname = "{PREFIX}{TAG}_{SUFFIX}.txt".format(PREFIX=prefix, TAG=tag,
                                                    SUFFIX=suffix)
        with open(self.get_file_path(src_idx, fname), "r") as txt_file:
            lines = txt_file.readlines()
            result_list = [float(line) for line in lines]
        return result_list


    def check_categorical(self, value_list, cat_cnt_thd):
        """
        chekc if a event is categorical based on its value list.
        """
        categorical_count = 0
        ratio_thd = 0.2 * len(value_list)
        cat_cnt_thd = np.min([cat_cnt_thd, len(value_list)])
        for value in list(value_list):
            if type(value) == str:
                categorical_count += 1

        if categorical_count < cat_cnt_thd or categorical_count < ratio_thd:
            return False
        return True
               

    def load_data_as_dict(self, src_idx, file_prefix="",
            low_freq_event_thd=0.05, low_freq_value_thd=0.05, cat_cnt_thd=4,
            simplify_patient_dict=True, mask_single_num_value_feature=False,
            zero_for_non_exist=False):
        """
        Extracting data from pre-processed dataset by George Chen;
        Transforming data with proper cleaning and processing;
        Loading data file into python dict.

        @params src_idx        Integer {0,1,2} Code choosing different dataset.
                               [src_idx]=0: "pancreatitis",
                               [src_idx]=1: "ich",
                               [src_idx]=2: "sepsis"
        @params file_prefix="" String. File prefix to choose cross validation
                               dataset or the complete train&test dataset.
                               [file_prefix]="": Use complete train & test
                               [file_prefix]="cross_val/x-xfold_x_": cross-val
        @params low_freq_event_thd=0.05 Integer [0, 1], If only less than
                               [low_freq_event_thd] * patients has a certain
                               event, drop this event for simplicity.
                               [low_freq_event_thd]=1: use all events.
        @params low_freq_value_thd=0.05 Integer [0, 1], If only less than
                               [low_freq_event_thd] * patients has a certain
                               value, drop this value for simplicity.
                               [low_freq_event_thd]=1: use all events.
        @params cat_cnt_thd=4  Integer [1, inf), If more than [cat_cnt_thd]
                               values of a event is str, then identify this
                               event as categorical.
                               [cat_cnt_thd]=1: use all events.

        @return patient_dict   dict. {patient_id: {processed_feature: value}}
        @return train_id_set   set. patient_id for training
        @return test_id_set    set. patient_id for training
        @return feature_list   list. all features to use
        @return cat_feature set. categorical feature
        """
       
        id_set = {}

        if self.verbose:
            print("[LOG]Reading train&test csv...")
        fname = file_prefix + "{}.csv"
        train_df = pd.read_csv(
            self.get_file_path(src_idx, fname.format("train")),
            header=None, names=["patientID", "event", "value"])
        test_df = pd.read_csv(
            self.get_file_path(src_idx, fname.format("test")),
            header=None, names=["patientID", "event", "value"])
        complete_df = pd.concat([train_df, test_df])
        patient_dict, all_na_event_set = self.from_df_to_dict(complete_df)
       
       
        if self.verbose:
            print("[LOG]Reading patient list, LoS, TUD...")
        for dataset_tag in ["train", "test"]:
            patient_id_list = self.read_text_file(src_idx, file_prefix,
                dataset_tag, "patients")
            los_value_list = self.read_text_file(src_idx, file_prefix,
                dataset_tag, "patient_ICU_LoS")
            tud_value_list = self.read_text_file(src_idx, file_prefix,
                dataset_tag, "patient_time_until_death_from_in_ICU")
            id_set[dataset_tag] = set(patient_id_list)
       
            for index, patient_id in enumerate(patient_id_list):
                patient_dict[patient_id]["LOS"] = [los_value_list[index]]
    #             patient_dict[patient_id]["TUD"] = [tud_value_list[index]]
                patient_dict[patient_id]["OUT"] = \
                    [int(los_value_list[index] != np.inf)]
                patient_dict[patient_id]["DIE"] = \
                    [int(tud_value_list[index] != np.inf)]

        if self.verbose:           
            print("[LOG]Identify one-hot encoded event, " + \
            "extract 'true event' and value.")
        # process all-na events (extract true event and value)
        na_new_event_set = set([])
        non_na_new_event_set = set([])

        for all_na_event in list(all_na_event_set):
            for patient_id in patient_dict.keys():
                if all_na_event in patient_dict[patient_id]:
                    event_tokens = all_na_event.split(":")
                    # If the all-na event cannot be divided as true event and
                    # value, directly delete this event now
                    # Notice that these events are likely to be meaningful if we
                    # take timestamp into consideration
                    if len(event_tokens) == 1:
                        patient_dict[patient_id][all_na_event] = ["done"]
                    # If the all-na event can be divided into multiple parts, we
                    # that the first few parts (other than the last one) will
                    # form the true event (feature) and the last part is the
                    # value.
                    # e.g. "microbiology:viral_culture:simplex_virus:herpes..."
                    # -> "microbiology:viral_culture:simplex_virus": "herpes..."
                    else:
                        patient_dict[patient_id].pop(all_na_event, None)
                        new_event = ":".join(event_tokens[:-1])
                        new_value = self.value_data_cleaning(event_tokens[-1],
                            new_event)
                        if new_value == 'na':
                            na_new_event_set.add(new_event)
                        else:
                            non_na_new_event_set.add(new_event)
                        patient_dict[patient_id][new_event].append(new_value)
        all_na_new_event_set = na_new_event_set - non_na_new_event_set
        for all_na_event in list(all_na_new_event_set):
            for patient_id in patient_dict.keys():
                if all_na_event in patient_dict[patient_id]:
                    patient_dict[patient_id][all_na_event] = ["exist"]

       
        if self.verbose:           
            print("[LOG]Count occurance for each event; Remove 'na' in value.")
        # count number of occurance for each event to filter out low-freq events
        # get all possible value for each event
        event_per_patient_count = defaultdict(int)
        event_value_dict = defaultdict(set)

        for patient_id in patient_dict.keys():
            ori_event_list = list(patient_dict[patient_id].keys())
            for event in ori_event_list:
                # Assume that 'na' does not provide any information.
                # Treat 'na' as missing value -> remove 'na'.
                ori_value_list = patient_dict[patient_id][event]
                value_list_without_na = [x for x in ori_value_list if x != 'na']

                # If the value list of this event of this patient has only 'na',
                # which was removed, then delete this event because there is no
                # information about this event.
                if len(value_list_without_na) == 0:
                    patient_dict[patient_id].pop(event, None)
                    continue

                # Update patient dict.
                patient_dict[patient_id][event] = value_list_without_na

                # Count occurance.
                event_per_patient_count[event] += 1

                # Get all possible value for the event.
                event_value_dict[event] |= set(value_list_without_na)

        if type(low_freq_event_thd) == float:
            feature_num_thd = int(len(patient_dict) * low_freq_event_thd)
        elif type(low_freq_event_thd) == int:
            feature_num_thd = low_freq_event_thd
        else:
            feature_num_thd = 1
        low_freq_events = set([])

        if self.verbose:
            print("[LOG]Feature filtering. Remove events that occurred" + \
            " for less than", feature_num_thd, "times.")

        if self.verbose:   
            print("[LOG]Number of events that ever occurred:",
            len(event_per_patient_count))
        # remove events that occurred for less than 0.05 * patient_number times
        for event in event_per_patient_count.keys():
            if event_per_patient_count[event] <= feature_num_thd:
                low_freq_events.add(event)
                event_value_dict.pop(event, None)
        if self.verbose:       
            print("[LOG]Number of events that occurred for more than" + \
            " {} times: {}".format(feature_num_thd, len(event_value_dict)))

        # list events that only has one value and process based on rules:
       
        # for categorical value, if the value itself does make sense, leave the
        # event as its current status, as during one-hot encoding this will
        # automatically result in a dummy feature indicating 1 or 0.
        # for value that does not make sense, e.g. "see comment", delete the
        # event.
       
        # for numerical value, this is problematic because this may result in
        # wrong inputation when fill in the missing value. Currently create a
        # mask feature to indicate the existance of this feature, and manually
        # fill in missing value (with 0).

        event_with_one_value = {}
        for event in list(event_value_dict.keys()):
            if event in set(["OUT", "LOS", "DIE", "TUD"]):
                continue
            if len(event_value_dict[event]) == 1:
                value = list(event_value_dict[event])[0]
                if type(value) == float or type(value) == int:
                    if value == 0:
                        event_with_one_value[event] = "mask"
                    if mask_single_num_value_feature:
                        event_with_one_value[event] = "mask"
                elif value == "see_comments":
                    event_with_one_value[event] = "delete"

        for patient_id in patient_dict.keys():
            for event in event_with_one_value:
                if event_with_one_value[event] == "mask":
                    new_event = event + "_existance_flag"
                    new_value = float(event in patient_dict[patient_id])
                    patient_dict[patient_id][new_event] = [new_value]
                    event_value_dict[new_event].add(new_value)

                    event_value_dict.pop(event, None)
                    patient_dict[patient_id].pop(event, None)
                elif event_with_one_value[event] == "delete":
                    event_value_dict.pop(event, None)
                    patient_dict[patient_id].pop(event, None)
               
        # identify categorical / numerical event
        if self.verbose:
            print("[LOG]Identify categorical event.")
        all_events = event_value_dict.keys()
        cat_events = set([])

        for event in all_events:
            # Get list of categorical event for ont-hot in the next step   
            if self.check_categorical(event_value_dict[event], cat_cnt_thd):
                cat_events.add(event)

        if type(low_freq_value_thd) == float:
            value_num_thd = int(len(patient_dict) * low_freq_value_thd)
        elif type(low_freq_value_thd) == int:
            value_num_thd = low_freq_value_thd
        else:
            value_num_thd = 1

        # count categorical event's value count
        cat_value_count = defaultdict(lambda: Counter())
        for patient_id in patient_dict.keys():
            for event in cat_events:
                if event not in patient_dict[patient_id]:
                    continue
                cat_value_count[event].update(patient_dict[patient_id][event])

        cat_value_ref = defaultdict(dict)
        for event in cat_events:
            for value in cat_value_count[event]:
                if cat_value_count[event][value] < value_num_thd:
                    cat_value_ref[event][value] = "combine_value"
                else:
                    cat_value_ref[event][value] = value

        if self.verbose:
            print("[LOG]One-hot encoding categorical event. " + \
                "Scalarize numerical.")
        # for categorical event, ont-hot for each patient
        cat_feature_count = Counter()
        for patient_id in patient_dict.keys():
            for event in all_events:
                if event not in patient_dict[patient_id]:
                    continue
                value_list = patient_dict[patient_id][event]
                if event in cat_events: # categorical event
                    patient_dict[patient_id].pop(event, None)
                    for value in set(value_list):
                        trans_value = cat_value_ref[event][value]
                        new_event = event + "__" + str(trans_value)
                        patient_dict[patient_id][new_event] = 1
                        cat_feature_count[new_event] += 1
                else: # numerical event
                    true_value_list = [x for x in value_list if type(x) != str]
                    if len(true_value_list) == 0:
                        patient_dict[patient_id].pop(event, None)
                    else:
                        patient_dict[patient_id][event] = \
                            self.scalarize_numerical_value_list(true_value_list)

        if self.verbose:
            print("[LOG]Value filtering. Remove values that occurred" + \
            " for less than", value_num_thd, "times.")
        cat_feature = set([])
        for feature in cat_feature_count:
            if cat_feature_count[feature] > value_num_thd:
                cat_feature.add(feature)
                # manually set the dummy feature as 0 if a patient does not
                # have this event-value pair.
                if zero_for_non_exist:
                    for patient_id in patient_dict:
                        if feature not in patient_dict[patient_id]:
                            patient_dict[patient_id][feature] = 0

        feature_set = (all_events - cat_events) | cat_feature

        if self.verbose:
            print("[LOG]After one-hot encoding, number of feature:",
              len(feature_set))
        if self.verbose:
            print("[LOG]Among them, number of categorical feature:",
              len(cat_feature))

        if self.verbose:
            print("[LOG]Simplify output dict and check correctness.")
        # check correct
        for patient_id in patient_dict.keys():
            if simplify_patient_dict:
                feature_delete = patient_dict[patient_id].keys() - feature_set
                for feature in feature_delete:
                    del patient_dict[patient_id][feature]
            # for feature in feature_set:
            #     if feature in patient_dict[patient_id]:
            #         value = patient_dict[patient_id][feature]
            #         assert type(value) != str and type(value) != list
            #         assert not math.isnan(value)

        train_id_list = list(id_set["train"])
        test_id_list = list(id_set["test"])


        return patient_dict, list(feature_set), train_id_list, test_id_list


    def get_diseases_list(self):
        return self.SOURCES
