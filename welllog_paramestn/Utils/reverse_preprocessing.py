import torch


def reverse_preproc(proc_info, name, predicted):
    for key in proc_info:
        if name != key:
            continue
        for i in reversed(range(len(proc_info[key]))):
            if proc_info[key][i][0] == "mn":
                min_value = float(proc_info[key][i][1] if proc_info[key][i][1][-1].isdigit() else proc_info[key][i][1][:-1])
                max_value = float(proc_info[key][i][2] if proc_info[key][i][2][-1].isdigit() else proc_info[key][i][2][:-1])
                predicted *= (max_value - min_value)
                predicted += min_value
            elif proc_info[key][i][0] == "log":
                predicted = torch.pow(10, predicted)
            elif proc_info[key][i][0] == "sd":
                mean_value = float(proc_info[key][i][1] if proc_info[key][i][1][-1].isdigit() else proc_info[key][i][1][:-1])
                std_value = float(proc_info[key][i][2] if proc_info[key][i][2][-1].isdigit() else proc_info[key][i][2][:-1])
                predicted *= std_value
                predicted += mean_value

    return predicted
