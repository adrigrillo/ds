from src.problems.weighted_completion_time import schedule_job_chains


def test_schedule_sample_case():
    weights_chain_1 = [6, 18, 12, 8]
    weights_chain_2 = [8, 17, 18]
    time_chain_1 = [3, 6, 6, 5]
    time_chain_2 = [4, 8, 10]
    chain_1 = [weights_chain_1, time_chain_1]
    chain_2 = [weights_chain_2, time_chain_2]
    chains = [chain_1, chain_2]
    result = schedule_job_chains(chains)
    assert result == [(0, 1), (1, 1), (0, 0), (1, 0), (0, 0)]
