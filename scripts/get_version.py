import toml

def get_version():
    return toml.load('pyproject.toml')['project']['version']

if __name__ == "__main__":
    print(get_version())
