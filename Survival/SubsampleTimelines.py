#!/usr/bin/env python
import argparse
import csv
import numpy as np
import os
import pathlib
import sys
from sklearn.model_selection import RepeatedKFold, ShuffleSplit


def create_directories(filename, input_is_dir=False):
    if input_is_dir:
        pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(os.path.dirname(filename)).mkdir(parents=True,
                                                      exist_ok=True)

def concatenate(pieces):
    new_pieces = []
    for piece in pieces:
        if ',' in piece:
            new_pieces.append('"' + piece + '"')
        else:
            new_pieces.append(piece)
    return ','.join(new_pieces)


def filter_data(csv_lines, use_first_in_ICU_time=True):
    birth_times = {}
    in_ICU_times = {}
    out_ICU_times = {}
    ICU_LoS = {}
    death_times = {}
    exclude_patients = set()

    # step 1: figure out birth, ICU entry, ICU exit, and death times
    chronological_csv_lines \
        = sorted(csv_lines, key=lambda pieces: float(pieces[1]))
    for pt, t, event, nondiscretized_val in chronological_csv_lines:
        if use_first_in_ICU_time:
            if event == 'inicu':
                if pt not in in_ICU_times:
                    in_ICU_times[pt] = float(t)
            elif event == 'outicu':
                if pt not in out_ICU_times:
                    out_ICU_times[pt] = float(t)
        else:
            # use whichever one is last by clobbering
            if event == 'inicu':
                in_ICU_times[pt] = float(t)
            elif event == 'outicu':
                out_ICU_times[pt] = float(t)

        # neither of the events below should have repeats...
        if event == 'birthed':
            birth_times[pt] = float(t)
        elif event == 'deceased':
            death_times[pt] = float(t)
        elif event == 'discharge_location:dead/expired,na':
            exclude_patients.add(pt)

    for pt in exclude_patients:
        if pt in in_ICU_times:
            del in_ICU_times[pt]
        if pt in out_ICU_times:  # how is this even possible if they're dead?
            del out_ICU_times[pt]

    # step 2: enumerate all the nondiscretized values per event
    new_csv_lines = []
    for pt, t, event, nondiscretized_val in chronological_csv_lines:
        if event == 'birthed' or event == 'date of birth':
            continue

        # only include patients who entered ICU and are not in the exclude list
        if pt in in_ICU_times:
            # only consider events before the patient `pt` entered ICU
            if float(t) < in_ICU_times[pt]:
                # for patients who died, only consider events before death
                if pt not in death_times \
                        or (pt in death_times and float(t) < death_times[pt]):
                    event = event.lower()
                    nondiscretized_val = nondiscretized_val.lower()
                    try:
                        float_val = float(nondiscretized_val)
                    except:
                        if event == 'chart:religion:na' \
                                or event == 'chart:religion:':
                            if nondiscretized_val == 'bud':
                                nondiscretized_val = 'buddhist'
                            elif nondiscretized_val == 'csc':
                                nondiscretized_val = 'christian scientist'
                            elif nondiscretized_val == 'epi':
                                nondiscretized_val = 'episcopalian'
                            elif nondiscretized_val == 'gor':
                                nondiscretized_val = 'greek orthodox'
                            elif nondiscretized_val == 'hin':
                                nondiscretized_val = 'hindu'
                            elif nondiscretized_val == 'jew':
                                nondiscretized_val = 'jewish'
                            elif nondiscretized_val == 'mus':
                                nondiscretized_val = 'muslim'
                            elif nondiscretized_val == 'nsp':
                                nondiscretized_val = 'not specified'
                            elif nondiscretized_val == 'otr':
                                nondiscretized_val = 'other'
                            elif nondiscretized_val == 'pro':
                                nondiscretized_val = 'protestant'
                            elif nondiscretized_val == 'rca':
                                nondiscretized_val = 'roman catholic'
                            elif nondiscretized_val == 'u':
                                nondiscretized_val = 'unknown'

                    new_csv_lines.append([pt, event, nondiscretized_val])

    # step 3: determine ages at ICU entry time
    for pt in in_ICU_times:
        if pt in birth_times:
            age_at_in_ICU \
                = (in_ICU_times[pt] - birth_times[pt]) / (60.*60.*24.*365.25)
            # warning: some entries are over 200 for people who are
            # sufficiently old (no they aren't actually over 200)
            new_csv_lines.append([pt, 'age at inicu', str(age_at_in_ICU)])

    # step 4: determine length of stay and time until death from ICU entry time
    time_until_death_from_in_ICU = {}
    for pt in in_ICU_times:
        if pt in out_ICU_times:
            ICU_LoS[pt] \
                = (out_ICU_times[pt] - in_ICU_times[pt]) \
                / (60.*60.*24.)
        else:
            # indicate that the survival time is censored
            ICU_LoS[pt] = np.inf

        if pt in death_times:
            time_until_death_from_in_ICU[pt] \
                = (death_times[pt] - in_ICU_times[pt]) \
                / (60.*60.*24.)
        else:
            # indicate that the survival time is censored
            time_until_death_from_in_ICU[pt] = np.inf

    return new_csv_lines, ICU_LoS, time_until_death_from_in_ICU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_timelines_csv_file')
    parser.add_argument('overall_subsample_factor', type=float)
    parser.add_argument('train_fraction', type=float)
    parser.add_argument('num_cross_val_folds', type=int)
    parser.add_argument('num_cross_val_repeats', type=int)
    parser.add_argument('random_seed', type=int)
    parser.add_argument('output_train_patients_file')
    parser.add_argument('output_train_csv_file')
    parser.add_argument('output_train_patient_ICU_LoS_file')
    parser.add_argument(
        'output_train_patient_time_until_death_from_in_ICU_file')
    parser.add_argument('output_test_patients_file')
    parser.add_argument('output_test_csv_file')
    parser.add_argument('output_test_patient_ICU_LoS_file')
    parser.add_argument(
        'output_test_patient_time_until_death_from_in_ICU_file')
    parser.add_argument('output_train_cross_val_dir')
    parser.add_argument('--random_permutation_cross_val',
                        action='store_true')
    args = parser.parse_args()

    input_csv_filename = args.input_timelines_csv_file
    overall_subsample_factor = args.overall_subsample_factor
    train_frac = args.train_fraction
    num_cv_repeats = args.num_cross_val_repeats
    num_folds = args.num_cross_val_folds

    rng = np.random.RandomState(args.random_seed)
    if args.random_permutation_cross_val:
        use_shuffle_split = True
    else:
        use_shuffle_split = False

    output_train_patients_filename = args.output_train_patients_file
    output_train_csv_filename = args.output_train_csv_file
    output_train_patient_ICU_LoS_filename \
        = args.output_train_patient_ICU_LoS_file
    output_train_patient_time_until_death_from_in_ICU_filename \
        = args.output_train_patient_time_until_death_from_in_ICU_file
        
    output_test_patients_filename = args.output_test_patients_file
    output_test_csv_filename = args.output_test_csv_file
    output_test_patient_ICU_LoS_filename \
        = args.output_test_patient_ICU_LoS_file
    output_test_patient_time_until_death_from_in_ICU_filename \
        = args.output_test_patient_time_until_death_from_in_ICU_file

    output_cross_val_dir = args.output_train_cross_val_dir

    create_directories(output_train_patients_filename)
    create_directories(output_train_csv_filename)
    create_directories(output_train_patient_ICU_LoS_filename)
    create_directories(
        output_train_patient_time_until_death_from_in_ICU_filename)
    create_directories(output_test_patients_filename)
    create_directories(output_test_csv_filename)
    create_directories(output_test_patient_ICU_LoS_filename)
    create_directories(
        output_test_patient_time_until_death_from_in_ICU_filename)

    print('Reading in data...')
    patients = set()
    with open(input_csv_filename) as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='"')
        lines = [row for row in csvreader]
        for pieces in lines[1:]:  # skip header (line 0)
            patient_id = int(float(pieces[0]))
            patients.add(patient_id)

    patients = sorted(list(patients), key=int)
    rng.shuffle(patients)
    if overall_subsample_factor < 1:
        target_num_patients = int(overall_subsample_factor*len(patients))
        patients = patients[:target_num_patients]
    num_train = int(train_frac*len(patients))

    train_patients_set = set(list(patients[:num_train]))
    test_patients_set = set(list(patients[num_train:]))
    train_lines = []
    test_lines = []
    with open(input_csv_filename) as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='"')
        lines = [row for row in csvreader]
        for pieces in lines[1:]:  # skip header (line 0)
            patient_id = int(float(pieces[0]))
            if pieces[1] == 'NA':
                print("NA timestamp. Skipping")
                continue
            if patient_id in train_patients_set:
                train_lines.append(pieces)
            elif patient_id in test_patients_set:
                test_lines.append(pieces)

    print('Filtering training data...')
    new_train_lines, train_ICU_LoS, train_time_until_death_from_in_ICU \
        = filter_data(train_lines)

    with open(output_train_csv_filename, 'w') as f:
        f.write('\n'.join([concatenate(pieces) for pieces in new_train_lines]))

    train_patients_that_enter_ICU = sorted(list(train_ICU_LoS.keys()))
    with open(output_train_patients_filename, 'w') as f:
        f.write('\n'.join(train_patients_that_enter_ICU))

    np.savetxt(output_train_patient_ICU_LoS_filename,
               np.array([train_ICU_LoS[pt]
                         for pt in train_patients_that_enter_ICU]))

    np.savetxt(output_train_patient_time_until_death_from_in_ICU_filename,
               np.array([train_time_until_death_from_in_ICU[pt]
                         for pt in train_patients_that_enter_ICU]))

    print('Filtering test data...')
    new_test_lines, test_ICU_LoS, test_time_until_death_from_in_ICU \
        = filter_data(test_lines)

    with open(output_test_csv_filename, 'w') as f:
        f.write('\n'.join([concatenate(pieces)
                           for pieces in new_test_lines]))

    test_patients_that_enter_ICU = sorted(list(test_ICU_LoS.keys()))
    with open(output_test_patients_filename, 'w') as f:
        f.write('\n'.join(test_patients_that_enter_ICU))

    np.savetxt(output_test_patient_ICU_LoS_filename,
               np.array([test_ICU_LoS[pt]
                         for pt in test_patients_that_enter_ICU]))

    np.savetxt(output_test_patient_time_until_death_from_in_ICU_filename,
               np.array([test_time_until_death_from_in_ICU[pt]
                         for pt in test_patients_that_enter_ICU]))

    # generate cross validation folds
    print('Assembling cross validation data...')
    create_directories(output_cross_val_dir, input_is_dir=True)
    rand_train_permutation \
        = rng.permutation(len(train_patients_that_enter_ICU))
    if use_shuffle_split:
        splitter = ShuffleSplit(n_splits=num_folds,
                                test_size=1/num_folds,
                                random_state=rng)
    else:
        splitter = RepeatedKFold(n_splits=num_folds,
                                 n_repeats=num_cv_repeats,
                                 random_state=rng)
    for fold_idx, (train, test) \
            in enumerate(splitter.split(train_patients_that_enter_ICU)):
        fold_train_patients \
            = [train_patients_that_enter_ICU[rand_train_permutation[idx]]
               for idx in train]
        fold_test_patients \
            = [train_patients_that_enter_ICU[rand_train_permutation[idx]]
               for idx in test]
        fold_train_patients_set = set(fold_train_patients)
        fold_test_patients_set = set(fold_test_patients)
        fold_test_patients = sorted(list(fold_test_patients_set))

        fold_file_prefix \
            = '%d-%dfold_%d' % (num_cv_repeats, num_folds, fold_idx)

        with open(os.path.join(output_cross_val_dir,
                               fold_file_prefix
                               + '_train.csv'), 'w') as f:
            f.write('\n'.join([concatenate(pieces)
                               for pieces in new_train_lines
                               if pieces[0] in fold_train_patients_set]))

        with open(os.path.join(output_cross_val_dir,
                               fold_file_prefix
                               + '_train_patients.txt'), 'w') as f:
            f.write('\n'.join(fold_train_patients))

        np.savetxt(os.path.join(output_cross_val_dir,
                                fold_file_prefix
                                + '_train_patient_ICU_LoS.txt'),
                   np.array([train_ICU_LoS[pt]
                             for pt in fold_train_patients]))

        np.savetxt(os.path.join(output_cross_val_dir,
                                fold_file_prefix
                                + '_train_patient'
                                + '_time_until_death'
                                + '_from_in_ICU.txt'),
                   np.array([train_time_until_death_from_in_ICU[pt]
                             for pt in fold_train_patients]))

        with open(os.path.join(output_cross_val_dir,
                               fold_file_prefix
                               + '_test.csv'), 'w') as f:
            f.write('\n'.join([concatenate(pieces)
                               for pieces in new_train_lines
                               if pieces[0] in fold_test_patients_set]))

        with open(os.path.join(output_cross_val_dir,
                               fold_file_prefix
                               + '_test_patients.txt'), 'w') as f:
            f.write('\n'.join(fold_test_patients))

        np.savetxt(os.path.join(output_cross_val_dir,
                                fold_file_prefix
                                + '_test_patient_ICU_LoS.txt'),
                   np.array([train_ICU_LoS[pt]
                             for pt in fold_test_patients]))

        np.savetxt(os.path.join(output_cross_val_dir,
                                fold_file_prefix
                                + '_test_patient'
                                + '_time_until_death'
                                + '_from_in_ICU.txt'),
                   np.array([train_time_until_death_from_in_ICU[pt]
                             for pt in fold_test_patients]))


if __name__ == '__main__':
    main()
