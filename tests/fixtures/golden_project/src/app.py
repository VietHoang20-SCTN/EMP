from service import Service


CONFIG = "golden"


def main():
    service = Service()
    return service.run(CONFIG)
