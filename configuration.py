from configparser import ConfigParser


def get_configuration(section=None):
    configuration = ConfigParser()
    configuration.read("configuration.ini")

    if section:
        return configuration[section]

    return configuration