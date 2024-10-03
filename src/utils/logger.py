# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """
    LOGROOT = os.path.join(os.path.join('./log/',config['model']),config['dataset'])
    # dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(LOGROOT):
        os.makedirs(LOGROOT)
    if "log_file_name" not in config:
        logfilename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    else:
        logfilename = config['log_file_name']

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    # comment following 3 lines and handlers = [sh, fh] to cancel file dump.
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        #handlers=[sh]
        handlers = [sh, fh]
    )


