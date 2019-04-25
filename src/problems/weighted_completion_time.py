from typing import Tuple, List


def calculate_ro_factor(jobs_chain: Tuple[List[int], List[int]]) -> Tuple[float, int]:
    """
    This method calculates the ro-factor of a given job chain composed
    by the weights and execution times.

    The ro-ratio is calculated saving the max of the cumulative sum of
    the weights divided by the cumulative sum of the execution time.

    :param jobs_chain: set of jobs with the weight and execution time
    :return: value of the ro-factor and the index of the job that generates it
    """
    ro_factor = 0
    job = 0
    cum_weight = 0
    cum_time = 0
    for i in range(len(jobs_chain[1])):
        cum_weight += jobs_chain[0][i]
        cum_time += jobs_chain[1][i]
        ratio = cum_weight / cum_time
        if ratio > ro_factor:
            ro_factor = ratio
            job = i
    return ro_factor, job


def schedule_job_chains(job_chains) -> List[Tuple[int, int]]:
    """
    Method that calculates the optimal schedule for the
    Total Weighted Completion Time problem for chains of jobs in a single
    machine.

    This set-up of the problem allows the change of the chain without all the
    jobs being fully completed.

    The result will return the chain and the number of jobs to execute, that
    after being executed are removed from the chain.

    :param job_chains: list of job chains
    :return: list with ordered tuples that indicates the chain to execute and
    the number of jobs that should be executed
    """
    schedule = list()
    while job_chains:
        max_ro_factor = 0
        last_job_chain = 0
        chain = 0
        # select the chain to be executed
        for i in range(len(job_chains)):
            ro_factor, job = calculate_ro_factor(job_chains[i])
            if ro_factor > max_ro_factor:
                max_ro_factor = ro_factor
                last_job_chain = job
                chain = i
        schedule.append((chain, last_job_chain))
        for i in range(len(job_chains[chain])):
            job_chains[chain][i] = job_chains[chain][i][last_job_chain + 1:]
        if not job_chains[chain][0]:
            job_chains.pop(chain)

    return schedule
