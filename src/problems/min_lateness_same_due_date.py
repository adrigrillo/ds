from collections import OrderedDict
from typing import Dict, Tuple, List


def calculate_lateness(time: int, job_process_time: int, due_date_job: int) -> int:
    """
    Method that calculates the tardiness of a job. `T_j = C_j - d_j`
    :param time: actual time of the execution
    :param job_process_time: time required to process the job
    :param due_date_job: due date to finish the job
    :return: lateness
    """
    time_after_job = time + job_process_time
    return time_after_job - due_date_job


def calculate_tardiness(time: int, job_process_time: int, due_date_job: int) -> int:
    """
    Method that calculates the lateness of a job. `L_j = max(T_j, 0) = max(C_j - d_j, 0)`

    :param time: actual time of the execution
    :param job_process_time: time required to process the job
    :param due_date_job: due date to finish the job
    :return: tardiness
    """
    lateness = calculate_lateness(time, job_process_time, due_date_job)
    return max(0, lateness)


def obtain_optimal_schedule_common_due_date(jobs: Dict[int, int], due_date: int) -> Tuple[List[int], int]:
    """
    Method that calculates the optimal execution of jobs with the same due date that
    minimizes the total tardiness of the set.

    :param jobs: dictionary with the jobs and its processing times
    :param due_date: due date for processing the jobs
    :return: list with the jobs to execute and total tardiness
    """
    sorted_jobs = OrderedDict(sorted(jobs.items(), key=lambda kv: kv[1]))
    tardiness = calculate_tardiness_jobs(sorted_jobs, due_date)
    return list(sorted_jobs.keys()), tardiness


def calculate_tardiness_jobs(jobs: Dict[int, int], due_date: int) -> int:
    """
    Method that calculates the tardiness of a set of jobs

    :param jobs: dictionary with the jobs and its processing times
    :param due_date: due date for processing the jobs
    :return: total tardiness
    """
    time = 0
    total_tardiness = 0
    for item, processing_time in jobs.items():
        lateness = calculate_tardiness(time, processing_time, due_date)
        total_tardiness += lateness
        time += processing_time
    return total_tardiness


def calculate_lateness_jobs(jobs: Dict[int, int], due_date: int) -> int:
    """
    Method that calculates the lateness of a set of jobs

    :param jobs: dictionary with the jobs and its processing times
    :param due_date: due date for processing the jobs
    :return: total lateness
    """
    time = 0
    total_lateness = 0
    for item, processing_time in jobs.items():
        lateness = calculate_lateness(time, processing_time, due_date)
        total_lateness += lateness
        time += processing_time
    return total_lateness
