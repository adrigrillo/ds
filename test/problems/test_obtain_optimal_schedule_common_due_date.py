from src.problems.min_lateness_same_due_date import obtain_optimal_schedule_common_due_date, calculate_tardiness_jobs


def test_same_due_date():
    jobs = {1: 2, 2: 6, 3: 4}
    jobs, tardiness = obtain_optimal_schedule_common_due_date(jobs=jobs, due_date=7)
    assert jobs == [1, 3, 2]
    assert tardiness == 5


def test_calculate_tardiness_jobs():
    jobs = {1: 2, 2: 4, 3: 6}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 5
    jobs = {1: 2, 3: 6, 2: 4}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 6
    jobs = {2: 4, 1: 2, 3: 6}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 5
    jobs = {2: 4, 3: 6, 1: 2}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 8
    jobs = {3: 6, 1: 2, 2: 4}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 6
    jobs = {3: 6, 2: 4, 1: 2}
    tardiness = calculate_tardiness_jobs(jobs=jobs, due_date=7)
    assert tardiness == 8