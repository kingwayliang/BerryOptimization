from Episode import Episode
from Agent import Agent
import numpy as np
from multiprocessing import shared_memory, Value, Process
from multiprocessing.managers import SharedMemoryManager
from time import time

numEpisode = 1200000
totalAttempts = 10
rewards = [1, 2]
probabilities = [0.9, 0.3]

learningRate: float = 0.0001

numProcs = 6


def spawnAgentAndTrain(qValues, smName, numEpisodes: int, learningRate: float):
    sm = shared_memory.SharedMemory(name=smName)
    q = np.ndarray(qValues.shape, dtype=qValues.dtype, buffer=sm.buf)

    agent = Agent(q, learningRate)

    e = 0
    while e < numEpisodes:
        e += 1
        episode = Episode(totalAttempts, rewards, probabilities)
        agent.run(episode)


def optimisticTrain(qValues, smName, perProcEpisodes):
    processes = [Process(target=spawnAgentAndTrain, args=(
        qValues, smName, perProcEpisodes, learningRate)) for _ in range(numProcs)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


# Locking is too slow
def synchronizedSpawnAgentAndTrain(qValues, smName, episodeCount: Value, numEpisodes):
    sm = shared_memory.SharedMemory(name=smName)
    q = np.ndarray(qValues.shape, dtype=qValues.dtype, buffer=sm.buf)

    agent = Agent(q, learningRate)

    while episodeCount.value < numEpisodes:
        episode = Episode(totalAttempts, rewards, probabilities)
        agent.run(episode)

        with episodeCount.get_lock():
            episodeCount.value += 1


def synchronizedTrain(qValues, smName):
    episodeCount = Value('i', 0)
    processes = [Process(target=synchronizedSpawnAgentAndTrain, args=(
        qValues, smName, episodeCount, numEpisode)) for _ in range(numProcs)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    print("Training for n = {0}, rewards of {1}, probabilities of {2}".format(
        totalAttempts, rewards, probabilities))
    print("Training for {0} episodes with learning rate {1} using {2} processes".format(
        numEpisode, learningRate, numProcs))

    # Dummy array for memory size
    # Initize Q values to be above expectation to achieve exploration
    qV = np.ones((totalAttempts, len(rewards))) * max(rewards)

    with SharedMemoryManager() as smm:
        qVSharedMem = smm.SharedMemory(qV.nbytes)
        qValues = np.ndarray(qV.shape, dtype=qV.dtype, buffer=qVSharedMem.buf)
        qValues[:, :] = qV[:, :]

        startTime = time()
        optimisticTrain(qValues, qVSharedMem.name, int(numEpisode / numProcs))
        print('Training took {} seconds'.format(time() - startTime))

    print("Q values")
    print(qValues)
    print("Policy")
    print(np.argmax(qValues, axis=1))
