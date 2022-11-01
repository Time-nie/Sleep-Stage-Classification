# Based on https://github.com/akaraspt/tinysleepnet

import glob
import os
import shutil
import pyedflib
import numpy as np
import argparse

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "MOVE": 5,
    "UNK": 6,
}


def preprocess(data_path, out_path='preprocessed', selected_ch='EEG Fpz-Cz'):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Read raw and annotation from EDF files
    psg_files = glob.glob(os.path.join(data_path, "*PSG.edf"))
    ann_files = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))
    psg_files.sort()
    ann_files.sort()

    for i in range(len(psg_files)):
        print(f"Processing {os.path.basename(psg_files[i])}")

        psg_reader = pyedflib.EdfReader(psg_files[i])
        ann_reader = pyedflib.EdfReader(ann_files[i])

        assert psg_reader.getStartdatetime() == ann_reader.getStartdatetime()

        epoch_duration = psg_reader.datarecord_duration
        if psg_reader.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2

        # Extract signal from the selected channel
        ch_names = psg_reader.getSignalLabels()
        select_ch_idx = -1
        for s in range(psg_reader.signals_in_file):
            if ch_names[s] == selected_ch:
                select_ch_idx = s
                break
        if select_ch_idx == -1:
            raise Exception("Channel not found.")
        sampling_rate = psg_reader.getSampleFrequency(select_ch_idx)
        n_epoch_samples = int(epoch_duration * sampling_rate)
        signals = psg_reader.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)

        # Sanity check
        n_epochs = psg_reader.datarecords_in_file
        if psg_reader.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_reader.readAnnotations()
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        x = x[select_idx]
        y = y[select_idx]

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]

        # Save
        filename = os.path.basename(psg_files[i]).replace("-PSG.edf", ".npz")
        np.savez(os.path.join(out_path, filename), x=x, y=y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data-1/sleep-edf-database-expanded-1.0.0/sleep-cassette",help="Path to the sleep-cassette folder in Sleep-EDF dataset."),
    parser.add_argument("--out_path", type=str, default="./preprocessed",
                        help="Directory where to save preprocessed data.")
    parser.add_argument("--selected_ch", type=str, default="EEG Fpz-Cz",
                        help="Name of the channel in the dataset.")
    args = parser.parse_args()
    preprocess(args.data_path, out_path=args.out_path, selected_ch=args.selected_ch)
