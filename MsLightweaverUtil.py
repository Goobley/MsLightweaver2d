import os
import os.path as path
import pickle
import signal, psutil

def test_timesteps_in_dir(path):
    filesInOutDir = [f for f in os.listdir(path) if f.startswith('Step_')]
    if len(filesInOutDir) > 0:
        print('Timesteps already present in output directory (%s), proceed? [Y/n]' % path)
        inp = input()
        if len(inp) > 0 and inp[0].lower() == 'n':
            raise ValueError('Data in output directory')

def optional_load_starting_context(folder, suffix=''):
    filename = 'StartingContext%s.pickle' % suffix
    if path.isfile(folder + filename):
        with open(folder + filename, 'rb') as pkl:
            result = pickle.load(pkl)
    else:
        result = None

    return result

# https://stackoverflow.com/a/45515052
def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)