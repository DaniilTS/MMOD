from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import simpy

fig, axs = plt.subplots(1)

AvQueueL = 'Av. queue length is: '
AvAmOfApp = 'Av. amount of applications in SMO: '
AvQueueTime = 'Av. time in queue is: '
AvSmoTime = 'Av. time in SMO is: '
AvAmBusyChannels = 'Av. amount of busy channels: '


class SMO:
    def __init__(self, env, channels_number, service_flow_rate, apps_flow_rate, waiting_flow_rate, max_queue_length):
        self.env = env
        self.service_flow_rate = service_flow_rate
        self.applications_flow_rate = apps_flow_rate
        self.queue_waiting_flow_rate = waiting_flow_rate
        self.max_queue_length = max_queue_length
        self.total_wait_times = []  # time in smo
        self.total_smo_list = []  # amount of apps in smo
        self.queue_times = []  # time of app in queue
        self.queue_list = []  # amount of apps in queue
        self.applications_done = []  # amount of processed apps for now
        self.applications_rejected = []  # same but rejected
        self.channel = simpy.Resource(env, channels_number)

    def app_processing(self):
        yield self.env.timeout(get_exponential(self.service_flow_rate))

    def app_waiting(self):
        yield self.env.timeout(get_exponential(self.queue_waiting_flow_rate))


def send_application(env, smo):
    queue_applications_amount = len(smo.channel.queue)
    processing_applications_amount = smo.channel.count
    smo.total_smo_list.append(queue_applications_amount + processing_applications_amount)
    smo.queue_list.append(queue_applications_amount)

    with smo.channel.request() as request:
        current_queue_len = len(smo.channel.queue)
        current_count_len = smo.channel.count
        if current_queue_len <= smo.max_queue_length:
            start_time = env.now
            smo.applications_done.append(current_queue_len + current_count_len)
            res = yield request | env.process(smo.app_waiting())
            smo.queue_times.append(env.now - start_time)

            if request in res:
                yield env.process(smo.app_processing())
                smo.total_wait_times.append(env.now - start_time)
        else:
            smo.applications_rejected.append(amount_of_channels + max_queue_length + 1)
            smo.queue_times.append(0)
            smo.total_wait_times.append(0)


def run_smo(env, smo):
    while True:
        yield env.timeout(get_exponential(smo.applications_flow_rate))
        env.process(send_application(env, smo))


def get_exponential(val):
    return np.random.exponential(1 / val)


def get_average_of(lst, text):
    average = np.array(lst).mean()
    print(text + str(average))
    return average


def find_empiric_probabilities(done_apps, rejected_apps, queue_times, total_wait_times, total_smo_list,
                               queue_list, amount_of_channels, max_queue_length, applications_flow_rate, service_flow_rate):
    total_applications_amount = len(done_apps) + len(rejected_apps)
    P = []
    for val in range(1, amount_of_channels + max_queue_length + 1):
        P.append(len(done_apps[done_apps == val]) / total_applications_amount)

    for index, p in enumerate(P):
        print('p{}: {}'.format(index, p))

    P_reject = len(rejected_apps) / total_applications_amount
    Q = 1 - P_reject  # prob that app is done
    A = applications_flow_rate * Q

    print('Empiric probability of rejection: {}'.format(P_reject))
    print('Empiric Q: {}'.format(Q))
    print('Empiric A: {}'.format(A))
    get_average_of(queue_list, AvQueueL)
    get_average_of(total_smo_list, AvAmOfApp)
    get_average_of(queue_times, AvQueueTime)
    get_average_of(total_wait_times, AvSmoTime)
    print(AvAmBusyChannels, Q * applications_flow_rate / service_flow_rate)

    axs.hist(total_wait_times, 100)
    axs.set_title('Awaiting time')


def find_theoretical_probabilities(amount_of_channels, max_queue_length, app_flow_rate, service_flow_rate, waiting_flow_rate):
    ro = app_flow_rate / service_flow_rate
    betta = waiting_flow_rate / service_flow_rate
    p0 = (sum([ro ** i / factorial(i) for i in range(amount_of_channels + 1)]) +
          (ro ** amount_of_channels / factorial(amount_of_channels)) *
          sum([ro ** i / (np.prod([amount_of_channels + t * betta for t in range(1, i + 1)]))
               for i in range(1, max_queue_length + 1)])) ** -1

    print('p0: {}'.format(p0))
    P = 0
    P += p0
    px = 0
    for i in range(1, amount_of_channels + 1):
        px = (ro ** i / factorial(i)) * p0
        P += px
        print('p{}: {}'.format(i, px))

    pn, pq = px, px

    for i in range(1, max_queue_length):
        px = (ro ** i / np.prod([amount_of_channels + t * betta for t in range(1, i + 1)])) * pn
        P += px
        if i < max_queue_length:
            pq += px
        print('P' + str(amount_of_channels + i) + ': ' + str(px))

    P = px
    Q = 1 - P
    A = Q * app_flow_rate
    L_q = sum([i * pn * (ro ** i) / np.prod([amount_of_channels + t * betta for t in range(1, i + 1)])
               for i in range(1, max_queue_length + 1)])
    L_pr = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, amount_of_channels + 1)]) + \
           sum([(amount_of_channels + index) * pn * ro ** index / np.prod(
               np.array([amount_of_channels + t * betta for t in range(1, index + 1)])) for index in
                range(1, max_queue_length + 1)])

    print('Theoretical probability of rejection: {}'.format(P))
    print('Theoretical Q: {}'.format(Q))
    print('Theoretical A: {}'.format(A))
    print(AvQueueL, L_q)
    print(AvAmOfApp, L_pr)
    print(AvQueueTime, Q * ro / app_flow_rate)
    print(AvSmoTime, L_pr / app_flow_rate)
    print(AvAmBusyChannels, Q * ro)


if __name__ == '__main__':
    amount_of_channels = 5  # n
    service_flow_rate = 5  # mu
    applications_flow_rate = 10  # lambda
    waiting_flow_rate = 1  # v
    max_queue_length = 1  # m

    env = simpy.Environment()
    smo = SMO(env, amount_of_channels, service_flow_rate, applications_flow_rate, waiting_flow_rate, max_queue_length)
    env.process(run_smo(env, smo))
    env.run(until=3000)

    print('Empiric Probabilities')
    find_empiric_probabilities(done_apps=np.array(smo.applications_done),
                               rejected_apps=np.array(smo.applications_rejected),
                               queue_times=np.array(smo.queue_times),
                               total_wait_times=np.array(smo.total_wait_times),
                               total_smo_list=np.array(smo.total_smo_list),
                               queue_list=np.array(smo.queue_list),
                               amount_of_channels=amount_of_channels,
                               max_queue_length=max_queue_length,
                               applications_flow_rate=applications_flow_rate,
                               service_flow_rate=service_flow_rate)

    print('\nTheoretical Probabilities')
    find_theoretical_probabilities(amount_of_channels=amount_of_channels,
                                   max_queue_length=max_queue_length,
                                   app_flow_rate=applications_flow_rate,
                                   service_flow_rate=service_flow_rate,
                                   waiting_flow_rate=waiting_flow_rate)
    plt.show()