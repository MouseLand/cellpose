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
    # check for passed arguments and run the command that was passed in the form --hook <hook_name>

    # Dictionary of hooks and their corresponding commands
    hooks = {
        "yapf-diff":
            'git diff -U0 --no-color --relative HEAD^ | yapf-diff -i --verbose --style "google"',
    }

    # Check if the hook is passed as an argument
    if sys.argv[1] == "--hook":
        hook = sys.argv[2]
        command = hooks.get(hook)
        print(hook, command)
        if command:
            print(f"Running command: {command}")
            run_command(command)
        else:
            print(f"Hook {hook} not found")
            sys.exit(1)
    else:
        print("Usage: python pre_commit_runner.py --hook <hook_name>")
        sys.exit(1)


if __name__ == "__main__":
    main()
