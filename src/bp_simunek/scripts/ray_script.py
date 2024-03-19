import time
import os
import ray
import scipy.stats as scs
from ray.util.actor_pool import ActorPool

os.environ["RAY_DEDUP_LOGS"] = "0"
ray.init()

@ray.remote
class remote_gen():
    """
    Analogous to flow123d solver. Contains a set and a get method, analogous to a set_parameters and get_observations in flow123d.
    """
    def set_gen(self, gen):
        # debug info
        print("init gen")
        print("mean", gen.mean())
        print("std", gen.std())
        self.gen = gen

    def rvs(self):
        random = self.gen.rvs(1)
        # sleep to occupy thread for longer
        time.sleep(2)
        # debug
        print(random)
        return random



@ray.remote
class remote_thread():
    """
    Analogous to a tinyDA remote chain. Main purpose is to call the remote_gen.
    """
    # access to generated values
    rvs = None
    # remote gen is supplied as a method parameter, can also be integrated into class itself
    def sample_random(self, gen):
        # debug
        print("submitting job")
        # call remote_gen
        job = gen.rvs.remote()
        # wait until the job is done
        ndone = job
        while ndone:
            _, ndone = ray.wait([ndone])
        # set value to be available for access outside of thread
        self.rvs = ray.get(job)

    # access results
    def get_rvs(self):
        return self.rvs


if __name__ == "__main__":
    # basically a tinyDA parallel chain
    # responsible for spawning remote chains
    # creation of and init of generators (flow123d solvers) should be handled outside of this

    # create pool of generators
    pool = [remote_gen.remote() for _ in range(0, 4)]
    # initialize generators cuz parametrized constructors dont seem to work
    jobs = []
    for idx, gen in enumerate(pool):
        jobid = gen.set_gen.remote((scs.norm(loc=idx*4, scale=0.01)))
        jobs.append(jobid)
    
    # wait for them all to finih
    while jobs:
        _, jobs = ray.wait(jobs)
    # create an actor pool from the generators
    pool = ActorPool(pool)
    # debug
    print("Pool initialized.")
    # initialize remote threads, aka tinyDA remote chains
    threads = [remote_thread.remote() for _ in range(4)]
    # debug
    print("Caller threads initialized.")
    # for every thread get an available generator and generate random values
    jobs = []
    for thread in threads:
        # available generator
        gen = pool.pop_idle()
        if gen:
            print(f"Submitting job for thread {thread}")
            print(f"Using generator {gen}")
            job = thread.sample_random.remote(gen)
            jobs.append(job)

    # wait for jobs to finish
    # should be per job likely, not waiting for all jobs
    # tinyDA handles that for us
    print("Jobs submitted.")
    while jobs:
        _, jobs = ray.wait(jobs)

    # retrieve results
    # also done by tinyDA
    rvs = ray.get([thread.get_rvs.remote() for thread in threads])
    print("Values retrieved.")
    print(rvs)

