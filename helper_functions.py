from definitions import *


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_str(s):
    try:
        str(s)
        return True
    except ValueError:
        return False


def get_bool(s):
    if is_int(s):
        if int(s) == 1:
            return True
        else:
            return False


def read_bool_parameter(line):
    value = line.strip()
    if value == 'True':
        return True
    elif value == 'False':
        return False
    else:
        exit(ERROR_PARAMS_FILE)


def get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value
