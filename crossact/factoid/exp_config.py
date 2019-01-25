__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import configparser

config = configparser.ConfigParser()


def read(settings_file):
    config.read(settings_file)
    for s in config.sections():
        print ('[' + s + ']')
        for x in config.options(s):
            print ('\t' + x + ':' + config.get(s, x))


def get_info():
    ret = ''

    for s in config.sections():
        ret += '[' + s + ']' + '\n'
        for x in config.options(s):
            ret += '\t' + x + ':' + config.get(s, x) + '\n'

    return ret


def get(_section, _option):
    if config.has_option(_section, _option):
        return config.get(_section, _option)
    return None


def set(_section, _option, _value):
    config.set(_section, _option, _value)
    print ('set', _section, _option, _value)
