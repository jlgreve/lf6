import logging
import os
import atexit


def get_logger(name, level='Info', file=None, criticals_dir=None):
    """
    Log with stream and optional file handler

    Arguments
    ---------
    name : string
        Name of the logger instance

    file : string (optional)
        path of the logfile to write to

    Returns
    -------
    Log object if it didnt already exist
    """
    level_dict = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}
    if not logging.Logger.manager.loggerDict.get(name):
        clog = logging.getLogger(name)
        clog.propagate = False
        clog.setLevel(level_dict[level.lower()])
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s :: %(message)s')
        clog.import_count = 0

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        clog.addHandler(sh)
        if file:
            fh = logging.FileHandler(file)
            fh.setFormatter(formatter)
            clog.addHandler(fh)

        if criticals_dir is not None:
            if not os.path.isdir(f'{criticals_dir}/'):
                os.makedirs(f'{criticals_dir}/')
            clog.criticals_file_path = f'{criticals_dir}/criticals.log'
        else:
            clog.criticals_file_path = 'criticals.log'

        # Always store criticals in log file for them to be accessible later
        with open(clog.criticals_file_path, 'w'):
            # Clear the contents of the log file at the beginning of each run
            pass
        fh = logging.FileHandler(clog.criticals_file_path)
        fh.setLevel(logging.CRITICAL)
        fh.setFormatter(formatter)
        clog.addHandler(fh)

    else:
        clog = logging.Logger.manager.loggerDict.get(name)
        clog.import_count += 1
        if clog.import_count == 1:
            atexit.register(lambda log: print(f'Logger {log.name} found by {log.import_count} function calls.'), clog)
        # clog.debug('Logger %s already exists.' % name)

    return clog
