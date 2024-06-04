import subprocess
import sys


def run_command(command):
    """Run a command platform independently and print the output. If there is an error, print the error and exit.
    """
    try:
        result = subprocess.run(command,
                                shell=True,
                                check=True,
                                capture_output=True,
                                text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(e.returncode)


def main():
    command = 'git diff -U0 --no-color --relative HEAD^ | yapf-diff -i --verbose --style "google"'
    print(f"Running command: {command}")
    run_command(command)


if __name__ == "__main__":
    main()
