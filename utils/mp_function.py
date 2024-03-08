import os
import numpy as np
import multiprocessing as mp


def get_process_num(segment_num: int) -> int:
    """
    Returns the number of process to be used for multiprocessing.
    Not recommended to use all the cores of the CPU.

    :param segment_num: total number of task(segment) to be preprocessed
    :return: process_num: number of process to be used
    """

    divisors = []
    for i in range(1, int(segment_num ** 0.5) + 1):
        if segment_num % i == 0:
            divisors.append(i)
            if i != segment_num // i:
                divisors.append(segment_num // i)
    available_divisor = [x for x in divisors if x < os.cpu_count()]

    return (
        int(os.cpu_count() * 0.6)
        if np.max(available_divisor) < os.cpu_count() // 2
        else np.max(available_divisor)
    )


def multi_process(target_function, case_list) -> None:
    """
    1. Split the patient_df into process_num
    2. Run target_function in parallel
    3. Concatenate the results from each process
    4. Check the integrity of the results
    5. Return ECG, FILE_NAME, SEX, AGE ( sex data is only used for integrity check )

    :param target_function: read_waveform_data
    :param patient_df: dataframe divided by process_num
    :param preprocess_cfg: preprocessing configuration
    :param debug_flag: flag for debugging
    :return:
    """
    process_num = get_process_num(len(case_list))
    print("process_num: {}".format(process_num))

    patients_per_process = np.array_split(case_list, process_num)

    with mp.Manager() as manager:

        workers = [
            mp.Process(
                target=target_function,
                args=(
                    process_i,
                    patients_per_process[process_i]
                ),
            )
            for process_i in range(process_num)
        ]

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        manager.shutdown()
