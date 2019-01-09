

def thread_start():
    import os
    import _thread
    import time
    if not os.path.exists('test'):
        os.makedirs('test')
    LOG_DIR = './test'
    get_ipython().system_raw('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(LOG_DIR))
    get_ipython().system_raw('./ngrok http 6006 &')

    def print_time(threadName, delay):
        count = 0
        while count < 2 * 60 * 60:
            time.sleep(delay)
            count += 1

    try:
        _thread.start_new_thread(print_time, ("Thread-1", 60,))
    except:
        print("Error: unable to start thread")
