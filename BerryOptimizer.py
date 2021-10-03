def berryOptimizer(n, r_low, r_high, p_low, p_high):
    actions = []
    expectedReward = []
    rNext = 0
    for _ in range(n):
        rHighProp = p_high * r_low + (1 - p_high) * rNext
        rHighReward = p_low * r_high + (1 - p_low) * rNext
        if rHighProp > rHighReward:
            actions.append('P')
            expectedReward.append(rHighProp)
        else:
            actions.append('R')
            expectedReward.append(rHighReward)
        rNext = expectedReward[-1]

    return actions, expectedReward


def catchRate(ball, berry, throw, medal, baseRate, CPM):
    # assume curve
    m = ball * 1.7 * berry * throw * medal
    return 1 - (1 - baseRate/(2*CPM)) ** m


lv20CPM = 0.6121573
lv25CPM = 0.667934

if __name__ == '__main__':
    # n = 14
    # r_low = 4
    # r_high = 7
    # p_high = catchRate(1, 2.5, 1.8, 1.4, 0.02, lv20CPM)
    # p_low = catchRate(1, 1, 1.8, 1.4, 0.02, lv20CPM)

    n = 10
    r_low = 1
    r_high = 2
    p_high = 0.9
    p_low = 0.3

    a, r = berryOptimizer(n, r_low, r_high, p_low, p_high)

    print('Optimization for n={0}, r_low={1}, r_high={2}, p_low={3:%}, p_high={4:%}'.format(
        n, r_low, r_high, p_low, p_high))
    for i in range(n):
        print(i+1, a[i], r[i])
