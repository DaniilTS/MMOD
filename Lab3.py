import random

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
    def __init__(self, env, channels_number, apps_flow_rate, max_queue_length, mu1, mu2):
        self.env = env
        self.mu1 = mu1
        self.mu2 = mu2
        self.applications_flow_rate = apps_flow_rate
        self.max_queue_length = max_queue_length
        self.total_smo_list = []  # amount of apps in smo
        self.total_wait_times = []
        self.queue_times = []  # time of app in queue
        self.queue_list = []  # amount of apps in queue
        self.applications_done = []  # amount of processed apps for now
        self.applications_rejected = []  # same but rejected
        self.channel = simpy.Resource(env, channels_number)

    def app_processing_1(self):
        yield self.env.timeout(np.random.exponential(1 / mu1))

    def app_processing_2(self):
        yield self.env.timeout(np.random.exponential(1 / mu2))


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
            yield request
            smo.queue_times.append(env.now - start_time)
            yield env.process(smo.app_processing_1())
            yield env.process(smo.app_processing_2())
            smo.total_wait_times.append(env.now-start_time)
        else:
            smo.applications_rejected.append(current_queue_len + max_queue_length + 1)


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


def find_empiric_probabilities(done_apps, rejected_apps, queue_times, total_smo_list, total_wait_times,
                               queue_list, amount_of_channels, max_queue_length, applications_flow_rate):

    total_applications_amount = len(done_apps) + len(rejected_apps)
    P = [len(done_apps[done_apps == val]) / total_applications_amount for val in range(1, amount_of_channels + max_queue_length + 1)]
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

    plt.hist(total_wait_times, 10)
    axs.set_title('Awaiting time')


def find_theoretical_probabilities(max_queue_length, app_flow_rate, service_flow_rate):
    pass
    # ro = app_flow_rate / service_flow_rate
    # p0 = (1 - ro) / (1 - (ro ** (max_queue_length + 2)))
    # p = [(ro ** i) * p0 for i in range(1, max_queue_length + 1)]
    # P_rejected = (ro ** (max_queue_length + 1)) * p0
    # Q = 1 - P_rejected
    # A = app_flow_rate * Q
    # L_q = (ro ** 2) * ((1 - (ro ** max_queue_length) * (max_queue_length * (1 - ro) + 1)) / ((1 - ro) ** 2)) * p0
    # L_pr = p[0] + L_q
    # av_time_in_smo = L_pr / app_flow_rate
    # av_time_in_queue = L_q / app_flow_rate
    #
    # print('p 0:', p0)
    # for i, el in enumerate(p):
    #     print('p', i + 1, ':', el)
    # print('Empiric probability of rejection: {}'.format(P_rejected))
    # print('Empiric Q: {}'.format(Q))
    # print('Empiric A: {}'.format(A))
    # print(AvQueueL, L_q)
    # print(AvAmOfApp, L_pr)
    # print(AvQueueTime, av_time_in_queue)
    # print(AvSmoTime, av_time_in_smo)
    # print(AvAmBusyChannels, Q * ro)


# 25. Имеется одноканальная СМО с очередью, ограниченной числом мест R = 2.
# На вход СМО поступает простейший по ток заявок с интенсивностью X.
# Время обслуживания распределено по обобщенному закону Эрланга с параметрами k, X2.
# Найти вероятности состояний СМО и характеристики эффективности СМО
if __name__ == '__main__':
    amount_of_channels = 1  # n
    mu1 = 6
    mu2 = 12
    apps_flow_rate = 2
    max_queue_length = 2

    env = simpy.Environment()
    smo = SMO(env, amount_of_channels, apps_flow_rate, max_queue_length, mu1, mu2)
    env.process(run_smo(env, smo))
    env.run(until=20000)

    print('Empiric Probabilities')
    find_empiric_probabilities(done_apps=np.array(smo.applications_done),
                               rejected_apps=np.array(smo.applications_rejected),
                               queue_times=np.array(smo.queue_times),
                               total_smo_list=np.array(smo.total_smo_list),
                               total_wait_times=np.array(smo.total_wait_times),
                               queue_list=np.array(smo.queue_list),
                               amount_of_channels=amount_of_channels,
                               max_queue_length=max_queue_length,
                               applications_flow_rate=apps_flow_rate)

    # print('\nTheoretical Probabilities')
    # find_theoretical_probabilities(max_queue_length=max_queue_length,
    #                                app_flow_rate=apps_flow_rate,
    #                                service_flow_rate=service_flow_rate)

    plt.show()
